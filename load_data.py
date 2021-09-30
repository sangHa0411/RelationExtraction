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
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sen_data = []
  for s, i, j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    sub_info=eval(i)
    obj_info=eval(j)
    
    subject_entity.append(sub_info['word'])
    object_entity.append(obj_info['word'])
        
    sub_type = 'SUB'
    sub_start = sub_info['start_idx']
    sub_end = sub_info['end_idx']
    obj_type = 'OBJ'
    obj_start = obj_info['start_idx']
    obj_end = obj_info['end_idx']
        
    sen = add_sep_tok(s, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type)
    sen_data.append(sen)

  out_dataset = pd.DataFrame({'id':dataset['id'], 
    'sentence':sen_data,
    'subject_entity':subject_entity,
    'object_entity':object_entity,
    'label':dataset['label'],}
  )
  
  return out_dataset

def load_data(dataset_dir):
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  return dataset

def add_sep_tok(sen, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type) :
  sub_sos = ' ['+sub_type+'_SOS] '
  sub_eos = ' ['+sub_type+'_EOS] '

  obj_sos = ' ['+obj_type+'_SOS] '
  obj_eos = ' ['+obj_type+'_EOS] '

  if sub_start < obj_start :
    sen = sen[:sub_start] + sub_sos + sen[sub_start:sub_end+1] + sub_eos + sen[sub_end+1:]
    obj_start += 22
    obj_end += 22
    sen = sen[:obj_start] + obj_sos + sen[obj_start:obj_end+1] + obj_eos + sen[obj_end+1:]
  else :
    sen = sen[:obj_start] + obj_sos + sen[obj_start:obj_end+1] + obj_eos + sen[obj_end+1:]
    sub_start += 22
    sub_end += 22
    sen = sen[:sub_start] + sub_sos + sen[sub_start:sub_end+1] + sub_eos + sen[sub_end+1:]
  return sen

# is preprocessor really useful
def tokenized_dataset(dataset, tokenizer, maxlen):
  entity_data = []
  sen_data = []
  for e01, e02, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    entity = e01 + ' [SEP] ' + e02

    entity_data.append(entity)
    sen_data.append(s)

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
