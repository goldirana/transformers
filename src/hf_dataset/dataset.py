from datasets import load_dataset, load_from_disk
import os
data_path = os.path.join(os.getcwd(), "datasets")
os.makedirs(data_path,exist_ok=False)

def get_banking_77():
    datasets = os.listdir(data_path)
    if 'banking77' not in datasets:
        dataset = load_dataset('legacy-datasets/banking77')
        dataset.save_to_disk(os.path.join(data_path, '/banking77'))
    else:
        dataset = load_from_disk(os.path.join(data_path, '/banking77'))
    return dataset