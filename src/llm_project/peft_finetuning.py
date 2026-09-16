from data_preperation import sentiment_mapping
import pandas as pd
import numpy as np
import torch
import json

task_prefix = "classify emotions: "

emotion_col = list(sentiment_mapping.keys())
#len_path can be changed to whatever language we got with other than english


#convert the one hot back to a list
def row_to_label(row) -> list:
    labels = []
    for e in emotion_col:
        if row[e] == 1:
            labels.append(e)
    return labels

#both language rows get loaded separately then concat into one set
#then shuffle them so the model don't get biased with all english rows and then swedish rows, we need to shuffle them
def bilingual_peft(en_path, lan_path, rand_int):
    en_df = pd.read_csv(en_path)
    lan_df = pd.read_csv(lan_path)
    combined = pd.concat([en_df, lan_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=rand_int).reset_index(drop=True)
    inputs = [task_prefix + e for e in combined["sentence"]]
    #get the list of active emotions and then convert to json and add to targets
    #json? model needs text(tokens)
    targets = []
    for _, row in combined.iterrows():
        labels = row_to_label(row)
        target = json.dumps(labels)
        targets.append(target)
    return inputs, targets

def main()
