from transformers import (MobileBertModel, MobileBertConfig, 
                          MobileBertForSequenceClassification, MobileBertTokenizer,
                          DistilBertConfig, DistilBertTokenizer, 
                          DistilBertForSequenceClassification)
import os
from src.hf_models.download_model import *
from src.config.configuration import ConfigurationManager

config_manager = ConfigurationManager()

model_path = os.path.join(os.getcwd(), "models")

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


def get_distilbert(task: str):
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
        model = DistilBertForSequenceClassification.from_pretrained(params.model_path)
        print("Model loaded successfully.")
    else:
        raise ValueError(f"Task {task} not recognized.")
    
    return model, tokenizer, config