import torch
import os
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from tqdm import tqdm

class ToxicDataset(Dataset):
    def __init__(self, responses, scores):
        self.responses = responses
        self.scores = torch.from_numpy(scores.to_numpy()).float()
        assert len(self.responses) == len(self.scores)

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        return self.responses[idx], self.scores[idx]