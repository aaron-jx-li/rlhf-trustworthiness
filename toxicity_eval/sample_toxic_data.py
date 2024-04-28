import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
csv_name = "../data/toxicity/anthropic/harmless/no_human_prompt_10000.csv"
df = pd.read_csv(csv_name)
sample_df = pd.DataFrame(columns=df.columns)
num_per_bin = 70
for i in range(5):
    selected_df = df[(df['chosen_score'] > i * 2 / 10) & (df['chosen_score'] < (i+1) * 2 / 10)]
    idx = np.random.choice(len(selected_df), size=num_per_bin, replace=False)
    sample_df = pd.concat([sample_df, selected_df.iloc[idx]], ignore_index=True)

# sorted_df = df.sort_values(by=['chosen_score'], ascending=False)
# print(sorted_df[['chosen_score', 'chosen']].head(n=10))
# plt.hist(sorted_df['chosen_score'])
# plt.savefig('./score_hist')
# toxic_df = sorted_df[sorted_df['chosen_score'] > 0.70]
# toxic_df = toxic_df[['prompt', 'chosen', 'prompt_score', 'chosen_score']]
# sample_df.to_csv('../data/toxicity/anthropic/harmless/probe_350.csv')
shuffled_indices = np.random.permutation(np.arange(len(sample_df)))
split_idx = int(len(shuffled_indices) * 0.7)
train_df = sample_df.iloc[shuffled_indices[:split_idx]]
test_df = sample_df.iloc[shuffled_indices[split_idx:]]
# train_df.to_csv('../data/toxicity/anthropic/harmless/probe_train.csv')
# test_df.to_csv('../data/toxicity/anthropic/harmless/probe_test.csv')
