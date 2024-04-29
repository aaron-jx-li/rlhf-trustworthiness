import pandas as pd
import os
import numpy as np
import re

# Modify if necessary
base_dir = "../data/toxicity/generations/"
model_type = 'ppo'
for filename in os.listdir(base_dir):
    if filename.endswith('_analyzed.csv'):
        csv_name = base_dir + filename
        print(filename)
        df = pd.read_csv(csv_name, lineterminator='\n')
        prob_data = df['log_prob'].to_numpy()
        prob_values = []
        for s in prob_data:
            if isinstance(s, float):
                prob_values.append(s)
            else:
                match = re.search(r'\((-?\d+\.\d+)', s)
                if match:
                    number = float(match.group(1))  # Convert the matched substring to float
                    prob_values.append(np.log(np.exp(number)*(100)))
        print(len(prob_values))
        print(np.nanmean(prob_values))

        
