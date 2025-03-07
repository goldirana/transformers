from dataclasses import dataclass

@dataclass
class DistilBert:
    model_path: str
    tokenizer_path: str
    config_path: str
    model_name: str
    
class DistilGPT2:
    model_path: str
    tokenizer_path: str
    config_path: str
    model_name: str
