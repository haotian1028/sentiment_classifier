import os
import re
import string
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from io import StringIO
from unicodedata import category
from bs4 import BeautifulSoup
from markdown import markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification, BertForMaskedLM,
    XLNetTokenizer, XLNetForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM,
    AlbertTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM,
    get_scheduler
)
from torch.optim import AdamW

MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODELS = [
    (BertForSequenceClassification, BertTokenizer, 'bert-base-cased'),
    (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
    (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
    (AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1')
]
MODEL_NAMES = ['bert', 'xlnet', 'roberta', 'albert']


def train_model(train_df, model_save_path, model_select=0):
    seed_torch(42)

    cur_model = MODELS[model_select]
    m_name = MODEL_NAMES[model_select]


    train_df['Polarity'] = train_df['Polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.Text.values
    labels = train_df.Polarity.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print(f'Training data shape: {input_ids.shape}, {attention_masks.shape}, {labels.shape}')


    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42)
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.1, random_state=42)


    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)


    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    print("Starting training...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

        train_acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")


        model.eval()
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
                val_labels.extend(b_labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")


        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")


    print("Final Model Performance on Validation Set:")
    print(classification_report(val_labels, val_predictions, digits=4))
    return model_save_path

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True