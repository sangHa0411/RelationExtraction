
import torch
import random
import collections
from torch.utils.data import Dataset, Subset

class RE_Dataset(Dataset):
  def __init__(self, pair_dataset, labels, val_ratio):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.val_ratio = val_ratio

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def split(self, k) :
    assert k < (1.0 / self.val_ratio)

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

      val_index = idx_list[k*(sample_size):(k+1)*sample_size]
      train_index = list(set(idx_list) - set(val_index))
            
      train_data.extend(train_index)
      val_data.extend(val_index)
        
    random.shuffle(train_data)
    random.shuffle(val_data)
        
    train_dset = Subset(self, train_data)
    val_dset = Subset(self, val_data)
    return train_dset, val_dset
