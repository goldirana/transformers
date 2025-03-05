from src.entity.config_entity import DistilBert
import os

model_path = os.path.join(os.getcwd(), "models")

class ConfigurationManager:
    def __init__(self):
        pass
    
    def get_distilbert_config(self) -> DistilBert:
        params = DistilBert(
            model_path=str(os.path.join(model_path,'/distilbert/model')),
            tokenizer_path=str(os.path.join(model_path,'/distilbert/tokenizer')),
            config_path=str(os.path.join(model_path,'/distilbert/config')),
            model_name='distilbert')
        return params
    