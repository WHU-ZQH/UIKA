from utils import *
import torch
from torch.utils.data import Dataset, DataLoader


class my_dataset(Dataset):

    def __init__(self, dataset, input_list):
        self.data = dataset
        self.len = len(dataset)
        self.input_list = input_list

    def __getitem__(self, index):
        return_value = []
        for input in self.input_list:
            if input in ['pw', 'mask']:
                data = torch.tensor(self.data[index][input], dtype=torch.float32)
            elif input in ['wids', 'tids', 'y', 'label', 'bert_token', 'bert_token_aspect']:
                data = torch.tensor(self.data[index][input], dtype=torch.long)
            else:
                data = self.data[index][input]
            return_value.append(data)
        return return_value

    def __len__(self):
        return self.len
