import pandas as pd
import yaml
from llm_project.peft_training import train_q_lora, train_lora

""" this scripts contains the training loop for LoRA, QLoRA fine-tuning from peft_training.py """

def run_training_loop(model, tokenizer, train_data, dev_data, config):
    """Run the training loop for LoRA or QLoRA fine-tuning."""
    train_data = load_data(config["train_path"], config["dev_path"], config["test_path"])
    if config.get("use_q_lora", False):
        
        train_q_lora(model, tokenizer, train_data, dev_data, config)
    else:
        train_lora(model, tokenizer, train_data, dev_data, config)
        

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_data(train_path: str, dev_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, development, and test datasets."""
    train_data = pd.read_json(train_path, lines=True)
    dev_data = pd.read_json(dev_path, lines=True)
    test_data = pd.read_json(test_path, lines=True)
    return train_data, dev_data, test_data
        
def run_inference(model, tokenizer, test_data: pd.DataFrame) -> list[str]:
    """Run inference on the test dataset."""
    predictions = []
    
    for index, row in test_data.iterrows():
        # 1. Build the prompt explicitly omitting the target "output"
        messages = [
            {"role": "system", "content": row.get("instruction", "")},
            {"role": "user", "content": row.get("input", "")}
        ]
        
        # 2. Apply template with add_generation_prompt=True to cue the assistant
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 3. Tokenize and move tensors to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 4. Generate the prediction
        outputs = model.generate(**inputs, max_new_tokens=50)
        
        # 5. Slice the output to exclude the prompt tokens, then decode
        input_length = inputs["input_ids"].shape[1]
        prediction = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        predictions.append(prediction)
        
    return predictions