import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# data_dirs = ['/n/home01/aaronli/DecodingTrust/data/toxicity/generations_0/toxic/benign/',
#              '/n/home01/aaronli/DecodingTrust/data/toxicity/generations_0/toxic/adversarial/',
#              #'/n/home01/aaronli/DecodingTrust/data/toxicity/generations_0/nontoxic/benign/',
#              #'/n/home01/aaronli/DecodingTrust/data/toxicity/generations_0/nontoxic/adversarial/'
#              ]
# model_names = [
#                "EleutherAI/pythia-70m", 
#                "EleutherAI/pythia-160m", 
#                "EleutherAI/pythia-410m",
#                "EleutherAI/pythia-2.8b", 
#                "skrishna/eleuther-pythia70m-hh-dpo",
#                "skrishna/eleuther-pythia160m-hh-dpo",
#                "skrishna/eleuther-pythia410m-hh-dpo",
#                "skrishna/eleuther-pythia2.8b-hh-dpo",
#                "skrishna/eleuther-pythia70m-hh-sft",
#                "skrishna/eleuther-pythia160m-hh-sft",
#                "skrishna/eleuther-pythia410m-hh-sft",
#                "skrishna/eleuther-pythia2.8b-hh-sft",
#                "usvsnsp/pythia-70m-ppo",
#                "usvsnsp/pythia-160m-ppo",
#                "usvsnsp/pythia-410m-ppo",
#                "usvsnsp/pythia-2.8b-ppo",
#                "EleutherAI/pythia-6.9b",
#                "skrishna/eleuther-pythia6.9b-hh-dpo",
#                "skrishna/eleuther-pythia6.9b-hh-sft",
#                "usvsnsp/pythia-6.9b-ppo"
#             ]
# out_rows = []
# column_names = ['text', 'score']
# out_df = pd.DataFrame(columns=column_names)
# for data_dir in data_dirs:
#     for model_name in model_names:
#         score_filename = data_dir + model_name.replace('/', '_') + '_analyzed.csv'
#         score_df = pd.read_csv(score_filename, lineterminator='\n')
        
#         for i in tqdm(range(len(score_df))):
#             text = score_df.iloc[i]['continuation']
#             score = score_df.iloc[i]['toxicity']
#             row_df = pd.DataFrame([[text, score]])
#             out_rows.append([text, score])
            
# print("Number of rows: ", len(out_rows))
# out_df = pd.DataFrame(out_rows, columns=column_names)
# out_df.to_csv("/n/home01/aaronli/DecodingTrust/data/toxicity/response_datasets/realtoxicity.csv")

# plt.hist(out_df['score'])
# plt.savefig('./score_hist_realtoxicity.png')

df = pd.read_csv("/n/home01/aaronli/DecodingTrust/data/toxicity/response_datasets/realtoxicity.csv", lineterminator='\n')
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

# train_df.to_csv("/n/home01/aaronli/DecodingTrust/data/toxicity/response_datasets/realtoxicity_train.csv")
# test_df.to_csv("/n/home01/aaronli/DecodingTrust/data/toxicity/response_datasets/realtoxicity_test.csv")

