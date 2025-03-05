from transformers import (MobileBertModel, MobileBertConfig, 
                          MobileBertForSequenceClassification, MobileBertTokenizer,
                          DistilBertConfig, DistilBertTokenizer, 
                          DistilBertForSequenceClassification)

def download_distilbert(task: str):
    config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    if task == "SequenceClassification":
        model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased')
    else:
        raise ValueError(f"Task {task} not recognized.")
    model.save_pretrained('/home/tess/work/deep_learning/transformers/models/distilbert/model')
    tokenizer.save_pretrained('/home/tess/work/deep_learning/transformers/models/distilbert/tokenizer')
    config.save_pretrained('/home/tess/work/deep_learning/transformers/models/distilbert/config')
    print("Model downloaded successfully.")