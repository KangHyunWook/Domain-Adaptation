from create_dataset import DEAP

from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, config):
        dataset = DEAP(config)
    
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    dataset=EEGDataset(config)


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        # collate_fn=collate_fn
    )

    return data_loader