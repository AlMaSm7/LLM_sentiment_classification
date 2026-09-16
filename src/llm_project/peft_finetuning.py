from data_preperation import sentiment_mapping
import pandas as pd
import numpy as np
import json
import argparse
from datasets import Dataset
#import transformers
import transformers

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

def main():
    #create a parser since we dont want to touch the source code
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-train", required=True)
    parser.add_argument("--sv-train", required=True)
    parser.add_argument("--output-dir", default="adapters/bilingual")
    parser.add_argument("--model-id", default="bigscience/mt0-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    inputs, targets = bilingual_peft(args.en_train, args.lan_train, args.seed)
    dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
