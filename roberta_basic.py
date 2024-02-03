import numpy as np
from datasets import Dataset
from utils import print_gpu_utilization, print_summary
import torch

seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

print("Init: ", print_gpu_utilization())

torch.ones((1,1)).cuda()
print("Kernels init:", print_gpu_utilization())

