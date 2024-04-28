import json
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import os
import argparse
import time
from toxic_dataset import ToxicDataset
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_probe import ToxicityProbe, ToxicityProbeClassify

type_to_dirs = [
    {'data_dir': '/n/home01/aaronli/DecodingTrust/data/toxicity/generations/toxic/benign/', 'out_dir': '/n/home01/aaronli/DecodingTrust/probe_models/realtoxicity/toxic/benign/'}, 
    {'data_dir': '/n/home01/aaronli/DecodingTrust/data/toxicity/generations/toxic/adversarial/', 'out_dir': '/n/home01/aaronli/DecodingTrust/probe_models/realtoxicity/toxic/adversarial/'}, 
    {'data_dir': '/n/home01/aaronli/DecodingTrust/data/toxicity/generations/nontoxic/benign/', 'out_dir': '/n/home01/aaronli/DecodingTrust/probe_models/realtoxicity/nontoxic/benign/'}, 
    {'data_dir': '/n/home01/aaronli/DecodingTrust/data/toxicity/generations/nontoxic/adversarial/', 'out_dir': '/n/home01/aaronli/DecodingTrust/probe_models/realtoxicity/nontoxic/adversarial/'}
    ]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_names = [
               "EleutherAI/pythia-70m", 
               "EleutherAI/pythia-160m", 
               "EleutherAI/pythia-410m",
               "EleutherAI/pythia-2.8b", 
               "skrishna/eleuther-pythia70m-hh-dpo",
               "skrishna/eleuther-pythia160m-hh-dpo",
               "skrishna/eleuther-pythia410m-hh-dpo",
               "skrishna/eleuther-pythia2.8b-hh-dpo",
               "skrishna/eleuther-pythia70m-hh-sft",
               "skrishna/eleuther-pythia160m-hh-sft",
               "skrishna/eleuther-pythia410m-hh-sft",
               "skrishna/eleuther-pythia2.8b-hh-sft",
               "usvsnsp/pythia-70m-ppo",
               "usvsnsp/pythia-160m-ppo",
               "usvsnsp/pythia-410m-ppo",
               "usvsnsp/pythia-2.8b-ppo",
               "EleutherAI/pythia-6.9b",
                "skrishna/eleuther-pythia6.9b-hh-dpo",
                "skrishna/eleuther-pythia6.9b-hh-sft",
                "usvsnsp/pythia-6.9b-ppo"
            ]

for prompt_type in range(len(type_to_dirs)):
    data_dir = type_to_dirs[prompt_type]['data_dir']
    out_dir = type_to_dirs[prompt_type]['out_dir']
    
    for name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models")
        base_model = GPTNeoXForCausalLM.from_pretrained(name, cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models").to(device)

        model = ToxicityProbeClassify(base_model)
        train_df = pd.read_csv(train_file_name)
        test_df = pd.read_csv(test_file_name)
        train_dataset = ToxicDataset(train_df['chosen'], train_df['chosen_score'])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataset = ToxicDataset(test_df['chosen'], test_df['chosen_score'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        optimizer_specs = [{'params':model.fc1.parameters(), 'lr':1e-3, 'weight_decay':1e-3},
                        {'params':model.fc2.parameters(), 'lr':1e-3, 'weight_decay':1e-3},
                        {'params':model.fc3.parameters(), 'lr':1e-3, 'weight_decay':1e-3}]
        optimizer = torch.optim.Adam(optimizer_specs)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        #criterion = torch.nn.CrossEntropyLoss().to(device)
        num_epochs = 30
        best_acc = 0
        best_epoch_num = 0

        layers_to_save = {key: value for key, value in model.state_dict().items() if key in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']}

        for j in range(num_epochs):
            total_training_loss = 0
            model.train()
            num_correct = 0
            for i, (prompt, score) in tqdm(enumerate(train_loader)):
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
                #score = score.unsqueeze(dim=0).to(device)
                if score > 0.5:
                    score = torch.tensor([1]).to(device)
                else:
                    score = torch.tensor([0]).to(device)
                model.to(device)
                pred = model(input_ids)
                # print(pred.shape, score.shape)
                if (pred[0][0] < pred[0][1] and score[0] > 0.5) or (pred[0][0] > pred[0][1] and score[0] < 0.5):
                    num_correct += 1
                loss = criterion(pred, score)
                total_training_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc = num_correct / len(train_dataset)
            print(f'Epoch {j} Training loss: {total_training_loss / i}')
            print(f"Training accuracy: {str(acc)}")

            model.eval()
            num_correct = 0
            for i, (prompt, score) in tqdm(enumerate(test_loader)):
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
                model.to(device)
                pred = model(input_ids)
                if (pred[0][0] < pred[0][1] and score[0] > 0.5) or (pred[0][0] > pred[0][1] and score[0] < 0.5):
                    num_correct += 1
            acc = num_correct / len(test_dataset)
            print(f"Test accuracy: {str(acc)}")

            if acc > best_acc:
                best_acc = acc
                best_epoch_num = j+1
                checkpoint = {key: value for key, value in model.state_dict().items() if key in layers_to_save}
                saved_acc = float("%.4f" % best_acc)
                torch.save(checkpoint, f"./probe_models/classification/{name.replace('/', '_')}_epoch{str(best_epoch_num)}_{str(saved_acc)}.pt")
        del model
        del tokenizer     