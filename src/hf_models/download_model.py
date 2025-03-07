from transformers import (MobileBertModel, MobileBertConfig, 
                          MobileBertForSequenceClassification, MobileBertTokenizer,
                          DistilBertConfig, DistilBertTokenizer, 
                          DistilBertForSequenceClassification,
                          GPT2Config, GPT2Tokenizer,
                          GPT2ForSequenceClassification)

import os
model_path = os.path.join(os.getcwd(), "models")
os.makedirs(model_path,exist_ok=True)


def download_distilbert(task: str):
    config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    if task == "SequenceClassification":
        model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased')
    else:
        raise ValueError(f"Task {task} not recognized.")
    model.save_pretrained(os.path.join(model_path, 'distilbert/model'))
    tokenizer.save_pretrained(os.path.join(model_path, 'distilbert/tokenizer'))
    config.save_pretrained(os.path.join(model_path, 'distilbert/config'))
    print("Model downloaded successfully.")
    
def download_distilgpt2(task: str):
    config = GPT2Config.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if task == "SequenceClassification":
        model = GPT2ForSequenceClassification.from_pretrained('distilgpt2')
    else:
        raise ValueError(f"Task {task} not recognized.")
    model.save_pretrained(os.path.join(model_path, 'distilgpt2/model'))
    tokenizer.save_pretrained(os.path.join(model_path, 'distilgpt2/tokenizer'))
    config.save_pretrained(os.path.join(model_path, 'distilgpt2/config'))
    print("Model downloaded successfully.")