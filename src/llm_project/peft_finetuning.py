from data_preperation import sentiment_mapping
import pandas as pd
import numpy as np
import json
import argparse
from datasets import Dataset
#import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

#run it by python peft_finetuning.py --en-train processed_data/en-annotated_processed.csv --lan-train processed_data/sv-projections_processed.csv

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
    combined = combined.dropna(subset=["sentence"]) 
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
    parser.add_argument("--lan-train", required=True)
    parser.add_argument("--output-dir", default="adapters/bilingual")
    parser.add_argument("--model-id", default="bigscience/mt0-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    inputs, targets = bilingual_peft(args.en_train, args.lan_train, args.seed)
    dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    input_len = [len(tokenizer(t)["input_ids"]) for t in dataset["input_text"]]
    target_len = [len(tokenizer(t)["input_ids"]) for t in dataset["target_text"]]
    max_source_length = int(np.percentile(input_len, 85))
    max_target_length = int(np.percentile(target_len, 90))

    def preprocess_function(sample,padding="max_len"):
        model_inputs=tokenizer(sample["input_text"], max_length = max_source_length, padding = "max_length", truncation=True)
        labels = tokenizer(text_target=sample["target_text"], max_length = max_target_length, padding = "max_length", truncation = True)
        labels["input_ids"] = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
                               for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    # when you're using a big model, you can quantisize the model  to save memory by using its
    # bit configuration in the parameter setting, that is, 'load_in_4bit=True' or 'load_in_8bit=True'
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, device_map="auto")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
 
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
if __name__ == "__main__":
    main()        
