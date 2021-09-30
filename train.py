import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (AutoTokenizer, 
  AutoConfig, 
  AutoModelForSequenceClassification, 
  Trainer, 
  TrainingArguments, 
  RobertaConfig, 
  RobertaTokenizer, 
  RobertaForSequenceClassification, 
  BertTokenizer
)

from load_data import *
import argparse
from pathlib import Path
import random
import wandb
from dotenv import load_dotenv

def klue_re_micro_f1(preds, labels):
  label_list = ['no_relation', 
    'org:top_members/employees', 
    'org:members',
    'org:product', 
    'per:title', 
    'org:alternate_names',
    'per:employee_of', 
    'org:place_of_headquarters', 
    'per:product',
    'org:number_of_employees/members', 
    'per:children',
    'per:place_of_residence', 
    'per:alternate_names',
    'per:other_family', 
    'per:colleagues', 
    'per:origin', 
    'per:siblings',
    'per:spouse', 
    'org:founded', 
    'org:political/religious_affiliation',
    'org:member_of', 
    'per:parents',
    'org:dissolved',
    'per:schools_attended', 
    'per:date_of_death', 
    'per:date_of_birth',
    'per:place_of_birth', 
    'per:place_of_death', 
    'org:founded_by',
    'per:religion']
  no_relation_label_idx = label_list.index("no_relation")
  label_indices = list(range(len(label_list)))
  label_indices.remove(no_relation_label_idx)
  return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
  labels = np.eye(30)[labels]
  score = np.zeros((30,))
  for c in range(30):
      targets_c = labels.take([c], axis=1).ravel()
      preds_c = probs.take([c], axis=1).ravel()
      precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
      score[c] = sklearn.metrics.auc(recall, precision)
  return np.average(score) * 100.0

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions
  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds)
  return {
    'micro f1 score': f1,
    'auprc' : auprc,
    'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  return num_label

def train(args):
  # -- Checkpoint Tokenizer
  MODEL_NAME = args.PLM
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # -- Device
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # -- Model
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config).to(device)

  # -- Resize Embedding
  special_tokens_dict = {'additional_special_tokens': ['[SUB_SOS]' ,
   '[SUB_EOS]',
   '[OBJ_SOS]' ,
   '[OBJ_EOS]']
  }
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))

  # -- Raw Data
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")
  train_label = label_to_num(train_dataset['label'].values)

  # -- Tokenizerd & Encoded Data
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, 256)

  # -- Dataset
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  train_dset, val_dset = RE_train_dataset.split()

  # -- Training Argument
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=3,              # number of total save model.
    save_steps=1000,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,                     # learning_rate
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,               # log saving step.
    evaluation_strategy=args.evaluation_strategy, # evaluation strategy to adopt during training
    eval_steps = 500,                # evaluation step.
    load_best_model_at_end = True,
    report_to='wandb'
  )

  # -- Trainer
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dset,            # training dataset
    eval_dataset=val_dset,               # evaluation dataset
    compute_metrics=compute_metrics      # define metrics function
  )

  # -- Training
  trainer.train()

  # -- Saving Model
  model.save_pretrained('./best_model')
  
def main(args):
  load_dotenv(dotenv_path=args.dotenv_path)
  WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
  wandb.login(key=WANDB_AUTH_KEY)

  wandb.init(
    entity="klue-level2-nlp-02",
    project="Relation-Extraction", 
    name=args.wandb_unique_tag,
    group=args.PLM)
  wandb.config.update(args)
  train(args)
  wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  ## Training Argument
  parser.add_argument('--dir_name', default='exp', help='model save at {SM_SAVE_DIR}/{name}')
  parser.add_argument('--PLM', type=str, default='monologg/koelectra-base-v3-discriminator', help='model type (default: monologg/koelectra-base-v3-discriminator)')
  parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
  parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size (default: 16)')
  parser.add_argument('--eval_batch_size', type=int, default=16, help='eval batch size (default: 16)')
  parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler (default: 500)')
  parser.add_argument('--weight_decay', type=float, default=1e-4, help='strength of weight decay (default: 1e-4)')
  parser.add_argument('--evaluation_strategy', type=str, default='steps', help='evaluation strategy to adopt during training, steps or epoch (default: steps)')

  # -- Seed
  parser.add_argument('--seed', type=int, default=2, help='random seed (default: 2)')

  # -- Wandb
  parser.add_argument('--dotenv_path', default='/opt/ml/wandb.env', help='input your dotenv path')
  parser.add_argument('--wandb_unique_tag', default='monologg/koelectra-base-v3-discriminator', help='input your wandb unique tag (default: monologg/koelectra-base-v3-discriminator)')  

  args = parser.parse_args()

  seed_everything(args.seed)   
  main(args)

