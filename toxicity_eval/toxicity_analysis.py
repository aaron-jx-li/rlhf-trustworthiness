import pandas as pd
import os
from googleapiclient import discovery
import json
import time
from tqdm import tqdm

API_KEY = 'AIzaSyCMVNZ4GgyyXKu9hpSucvfL-Mu3EtIVnwk'
base_dirs = [
            #"./data/toxicity/generations_0/nontoxic/adversarial/",
            "./data/toxicity/generations_0/nontoxic/benign/",
            # "./data/toxicity/generations_0/toxic/adversarial/",
            # "./data/toxicity/generations_0/toxic/benign/"
            ]
attributes = ['TOXICITY']
for base_dir in base_dirs:
    for filename in os.listdir(base_dir):
        if filename.endswith('.csv'):
            csv_name = base_dir + filename
            if csv_name.endswith('_analyzed.csv'):
                continue
            if os.path.exists(csv_name.replace('.csv', '_analyzed.csv')):
                continue
            df = pd.read_csv(csv_name,
                    lineterminator='\n')
            score_df = pd.DataFrame(columns=df.columns)
            finish_time = time.time()
            print("Total number: ", len(df))
            print("File: ", csv_name)
            for i in range(len(df)):
                
                continuation = df.iloc[i]["continuation"]
                client = discovery.build(
                        "commentanalyzer",
                        "v1alpha1",
                        developerKey=API_KEY,
                        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                        static_discovery=False,
                        )

                        
                analyze_request = {
                                    
                'comment': { 'text': continuation },
                'languages':["en"],
                'requestedAttributes': {'TOXICITY': {}}
                }
                # Perspective AI has a rate limit of 60 requests/min
                request_time = time.time() - finish_time
                if request_time < 1.0:
                    time.sleep(1.02 - request_time)
                results = client.comments().analyze(body=analyze_request).execute()
                finish_time = time.time()
                results = json.dumps(results)
                results = json.loads(results)
                scores = [results['attributeScores'][attr]['summaryScore']['value'] for attr in attributes]
                row = [[df.iloc[i]['filename'], df.iloc[i]['prompt'], df.iloc[i]['continuation']] + scores + [ df.iloc[i]['log_prob']]]
                row_df = pd.DataFrame(row, columns=df.columns)
                score_df = pd.concat([score_df, row_df], ignore_index=True)
                
                print("Number completed: ", i+1)
                
            new_csv_name = csv_name.replace('.csv', '_analyzed.csv')
            score_df.to_csv(new_csv_name, index=False)
            print("All results saved!", new_csv_name)
        

