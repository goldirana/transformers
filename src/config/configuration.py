from src.entity.config_entity import DistilBert, DistilGPT2
import os
from pathlib import Path
model_path = Path(os.path.join(os.getcwd(), "models"))

class ConfigurationManager:
    def __init__(self):
        pass
    
    def get_distilbert_config(self) -> DistilBert:
        params = DistilBert(
            model_path=str(Path(os.path.join(model_path,'distilbert/model'))),
            tokenizer_path=str(Path(os.path.join(model_path,'distilbert/tokenizer'))),
            config_path=str(Path(os.path.join(model_path,'distilbert/config'))),
            model_name='distilbert')
        print(model_path)
        return params
    
    def get_distilgpt2_config(self) -> DistilGPT2:
        params = DistilBert(
            model_path=str(Path(os.path.join(model_path,'distilgpt2/model'))),
            tokenizer_path=str(Path(os.path.join(model_path,'distilgpt2/tokenizer'))),
            config_path=str(Path(os.path.join(model_path,'distilgpt2/config'))),
            model_name='distilgpt2')
        return params
    
    
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    params = config_manager.get_distilbert_config()
    print(params.model_path)
    print(params.tokenizer_path)
    print(params.config_path)
    print(params.model_name)   
    print(model_path)