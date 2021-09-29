import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from preprocessor import *

class RE_Dataset(Dataset):
  def __init__(self, pair_dataset, labels, val_ratio=0.1):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.val_ratio = val_ratio

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def split(self) :
    data_size = len(self)
    index_map = collections.defaultdict(list)
    for idx in range(data_size) :
        label = self.labels[idx]
        index_map[label].append(idx)
            
    train_data = []
    val_data = []
        
    label_size = len(index_map)
    for label in range(label_size) :
      idx_list = index_map[label]    
      sample_size = int(len(idx_list) * self.val_ratio)

      val_index = random.sample(idx_list, sample_size)
      train_index = list(set(idx_list) - set(val_index))
            
      train_data.extend(train_index)
      val_data.extend(val_index)
        
    random.shuffle(train_data)
    random.shuffle(val_data)
        
    train_dset = Subset(self, train_data)
    val_dset = Subset(self, val_data)
    return train_dset, val_dset

def preprocessing_dataset(dataset):
  subject_entity = []
  object_entity = []
  for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = eval(i)['word']
    j = eval(j)['word']
    subject_entity.append(i)
    object_entity.append(j)

  out_dataset = pd.DataFrame({'id':dataset['id'], 
    'sentence':dataset['sentence'],
    'subject_entity':subject_entity,
    'object_entity':object_entity,
    'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  return dataset

def tokenized_dataset(dataset, tokenizer, maxlen):
  entity_data = []
  sen_data = []
  for e01, e02, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    entity = e01 + ' [SEP] ' + e02
    entity = preprocess_sen(entity)
    sen = preprocess_sen(s)

    entity_data.append(entity)
    sen_data.append(sen)

  tokenized_sentences = tokenizer(
    entity_data,
    sen_data,
    return_tensors="pt",
    truncation=True,
    padding='max_length',
    max_length=maxlen,
    add_special_tokens=True
  )
   
  return tokenized_sentences
