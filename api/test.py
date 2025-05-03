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
from api.train import *

def test_model(test_df, model_saved_path, model_select=0):

  MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'),
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
        ]
  MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
  seed_torch(42)

  cur_model=MODELS[model_select]
  m_name=MODEL_NAMES[model_select]

  tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

  begin=time.time()

  test_df['Polarity']=test_df['Polarity'].replace({
      'positive':1,
      'negative':2,
      'neutral':0})


  sentences = test_df.Text.values
  labels = test_df.Polarity.values

  input_ids = []
  attention_masks = []

  for sent in sentences:
      encoded_dict = tokenizer.encode_plus(
                          str(sent),
                          add_special_tokens = True,
                          max_length = MAX_LEN,
                          pad_to_max_length = True,
                          return_attention_mask = True,
                          return_tensors = 'pt',
                    )

      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])

  prediction_inputs = torch.cat(input_ids,dim=0)
  prediction_masks = torch.cat(attention_masks,dim=0)
  prediction_labels = torch.tensor(labels)

  prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

  model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
  model.load_state_dict(torch.load(model_saved_path))
# model.cuda()
  model.eval()

  predictions,true_labels=[],[]

  for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch

      with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits = outputs[0]

      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      predictions.append(logits)
      true_labels.append(label_ids)

  end=time.time()
  print('Prediction used {:.2f} seconds'.format(end - begin))

  flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  flat_true_labels = [item for sublist in true_labels for item in sublist]

  print("Accuracy of {} is: {}".format(m_name, accuracy_score(flat_true_labels,flat_predictions)))

  print(classification_report(flat_true_labels,flat_predictions))


  df_prediction = pd.DataFrame(flat_predictions, columns=['prediction_Polarity'])

  df_combined = pd.concat([test_df, df_prediction], axis=1)

  counts = df_combined['prediction_Polarity'].value_counts()
  print(counts)

  return df_combined