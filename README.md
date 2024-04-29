# Official Code For the Paper "More RLHF, More Trust? On The Impact of Human Preference Alignment On Language Model Trustworthiness"

## Overview

This repo contains the experiment code for the paper "More RLHF, More Trust? On The Impact of Human Preference Alignment On Language Model Trustworthiness". See our paper for details.

This project is organized around the following **five** primary areas of trustworthiness, including:
1. Toxicity
2. Stereotypical bias
3. Machine Ethics
4. Truthfulness
5. Privacy 

## Project Structure
This project is structured around subdirectories dedicated to each area of trustworthiness. Each subdir includes python scripts to reproduce our experiment results. Additionally, the data for each evaluation is included in the `data/` subdirectory.

```
.
├── README.md
├── data/
    ├── toxicity/
    ├── stereotype/
    ├── machine_ethics/
    ├── truthfulness/
    ├── privacy/
├── toxicity_eval/
├── stereotype_eval/
├── machine_ethics/
├── privacy/
├── ood/
├── machine_ethics/
└── fairness/
```


(⚠️WARNING: This paper contains model outputs that may be considered offensive.)

In general, python scripts end with `_exp.py` are used for language model inference, `_analysis.py` are used for calculating the specific benchmark values. 

## [Important] Candidate models
In this project, we mainly focus on the open source Pythia suite, and here are the links to the specific models hosted on **Huggingface**

- `Pythia 70M`: EleutherAI/pythia-70m
- `Pythia 160M`: EleutherAI/pythia-160m
- `Pythia 410M`: EleutherAI/pythia-410m
- `Pythia 2.8B`: EleutherAI/pythia-2.8b
- `Pythia 6.9B`: EleutherAI/pythia-6.9b

