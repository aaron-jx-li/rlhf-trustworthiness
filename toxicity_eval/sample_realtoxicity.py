import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Modify if needed
data_dirs = [
             '../data/toxicity/generations/toxic/benign/',
             '../data/toxicity/generations/toxic/adversarial/',
             '../data/toxicity/generations/nontoxic/benign/',
             '../data/toxicity/generations/nontoxic/adversarial/'
             ]

# Models to be evaluated
model_names = []
out_rows = []
column_names = ['text', 'score']
out_df = pd.DataFrame(columns=column_names)
for data_dir in data_dirs:
    for model_name in model_names:
        score_filename = data_dir + model_name.replace('/', '_') + '_analyzed.csv'
        score_df = pd.read_csv(score_filename, lineterminator='\n')
        
        for i in tqdm(range(len(score_df))):
            text = score_df.iloc[i]['continuation']
            score = score_df.iloc[i]['toxicity']
            row_df = pd.DataFrame([[text, score]])
            out_rows.append([text, score])
            

df = pd.DataFrame(out_rows, columns=column_names)

df = pd.read_csv("../data/toxicity/response_datasets/realtoxicity.csv", lineterminator='\n')
print(len(df))
sample_df = pd.DataFrame(columns=df.columns)
num_per_bin = 5000
for i in range(5):
    selected_df = df[(df['score'] > i * 2 / 10) & (df['score'] < (i+1) * 2 / 10)]
    print(len(selected_df), i)
    idx = np.random.choice(len(selected_df), size=num_per_bin, replace=False)
    sample_df = pd.concat([sample_df, selected_df.iloc[idx]], ignore_index=True)
shuffled_indices = np.random.permutation(np.arange(len(sample_df)))
split_idx = int(len(shuffled_indices) * 0.5)
train_df = sample_df.iloc[shuffled_indices[:split_idx]]
test_df = sample_df.iloc[shuffled_indices[split_idx:]]

train_df.to_csv("../data/toxicity/response_datasets/realtoxicity_train.csv")
test_df.to_csv("../data/toxicity/response_datasets/realtoxicity_test.csv")

