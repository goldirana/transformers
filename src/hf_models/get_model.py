from transformers import (MobileBertModel, MobileBertConfig, 
                          MobileBertForSequenceClassification, MobileBertTokenizer,
                          DistilBertConfig, DistilBertTokenizer, 
                          DistilBertForSequenceClassification,
                          GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer)
import os
from src.hf_models.download_model import *
from src.config.configuration import ConfigurationManager

config_manager = ConfigurationManager()
# get the current file path


model_path = os.path.join(os.getcwd(), "models")
os.makedirs(model_path, exist_ok=True)

def check_model_present_in_local(model_name):
    models = os.listdir(model_path)
    if model_name in models:
        return True
    else:
        return False

def get_mobilebert(task: str):
    config = MobileBertConfig.from_pretrained('/home/tess/work/deep_learning/transformers/models/mobilebert/config')
    tokenizer = MobileBertTokenizer.from_pretrained('/home/tess/work/deep_learning/transformers/models/mobilebert/tokenizer')
    if task == "SequenceClassification":
        model = MobileBertForSequenceClassification.from_pretrained('/home/tess/work/deep_learning/transformers/models/mobilebert/model')
    else:
        raise ValueError(f"Task {task} not recognized.")
    return model, tokenizer, config


def get_distilbert(task: str,  **kwargs):
    _ = check_model_present_in_local('distilbert')
    if _ == False:
        try:
            # download the model
            download_distilbert(task)
        except:
            raise ValueError("Model not found in Hugging Face model hub.")
    params = config_manager.get_distilbert_config()
    
    config = DistilBertConfig.from_pretrained(params.config_path)
    tokenizer = DistilBertTokenizer.from_pretrained(params.tokenizer_path)
    if task == "SequenceClassification":
        model = DistilBertForSequenceClassification.from_pretrained(params.model_path,
                                                                    
                                                                    ignore_mismatched_sizes=True,
                                                                    **kwargs
                                                                    )
        print("Model loaded successfully.")
    else:
        raise ValueError(f"Task {task} not recognized.")
    
    return model, tokenizer, config

def get_distilgpt2(task: str, **kwargs):
    _ = check_model_present_in_local('distilgpt2')
    if _ == False:
        try:
            # download the model
            download_distilgpt2(task)
        except:
            raise ValueError("Model not found in Hugging Face model hub.")
    params = config_manager.get_distilgpt2_config()
    
    config = GPT2Config.from_pretrained(params.config_path)
    tokenizer = GPT2Tokenizer.from_pretrained(params.tokenizer_path)
    if task == "SequenceClassification":
        model = GPT2ForSequenceClassification.from_pretrained(params.model_path,
                                                                    ignore_mismatched_sizes=True,
                                                                    cache_dir="./fresh_cache",
                                                                    **kwargs
                                                                    )
        print("Model loaded successfully.")
    else:
        raise ValueError(f"Task {task} not recognized.")
    
    return model, tokenizer, config