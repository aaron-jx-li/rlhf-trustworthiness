import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from toxic_dataset import ToxicDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class ToxicityProbeClassify(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc1 = torch.nn.Linear(base_model.config.hidden_size, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 2)   
        self.embedding_out = None

    def forward(self, x):
        out = self.base_model(x, output_hidden_states=True).hidden_states
        self.embedding_out = out[0]
        out = out[1][:, -1, :]
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
