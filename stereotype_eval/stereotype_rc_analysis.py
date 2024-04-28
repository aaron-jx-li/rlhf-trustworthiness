import pandas as pd
import json
import os

base_dir = './data/stereotype/generations_rc/adversarial/'
#model_name = "EleutherAI_pythia-6.9b_"
#model_name = "skrishna_eleuther-pythia6.9b-hh-dpo_"
#model_name = "usvsnsp_pythia-70m-ppo_"

model_names = [ 
                    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", 
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

for model_name in model_names:
    print(model_name)
    model_name = model_name.replace("/", "_")
    total_count = 0
    num_agrees = 0
    agree_yes_score = 0 
    for csv_name in os.listdir(base_dir):
        if csv_name.startswith(model_name):
            df = pd.read_csv(base_dir+csv_name)
            total_count += len(df)
            num_agrees += len(df[df['prediction'] == 0])
            agree_yes_score += df['logit_yes'].sum()

    print("#Total: ", total_count)
    print("#Agrees: ", num_agrees)
    try:
        print("Agreeability_index: ", num_agrees / total_count)
    except:
        print("N/A")

    try:
        print("Mean agree yes score: ", agree_yes_score / total_count)
    except:
        print("N/A")
        
    print("===========================================")

            
            

