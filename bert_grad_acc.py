import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForSequenceClassification
from profile_utils import print_gpu_utilization, print_summary
import torch
logging.set_verbosity_error()

seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print("After Model init:")
print_gpu_utilization()

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)