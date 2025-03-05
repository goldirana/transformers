from datasets import load_dataset, load_from_disk
import os


def get_banking_77():
    datasets = os.listdir("/home/tess/work/deep_learning/transformers/datasets")
    if 'banking77' not in datasets:
        dataset = load_dataset('legacy-datasets/banking77')
        dataset.save_to_disk('/home/tess/work/deep_learning/transformers/datasets/banking77')
    else:
        dataset = load_from_disk('/home/tess/work/deep_learning/transformers/datasets/banking77')
    return dataset