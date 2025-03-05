from src.entity.config_entity import DistilBert


class ConfigurationManager:
    def __init__(self):
        pass
    
    def get_distilbert_config(self) -> DistilBert:
        params = DistilBert(
            model_path='/home/tess/work/deep_learning/transformers/models/distilbert/model',
            tokenizer_path='/home/tess/work/deep_learning/transformers/models/distilbert/tokenizer',
            config_path='/home/tess/work/deep_learning/transformers/models/distilbert/config',
            model_name='distilbert')
        return params
    