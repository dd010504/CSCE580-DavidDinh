import os
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Load IMDB Dataset 
print("Downloading IMDB dataset...")
dataset = load_dataset("imdb")

# Save raw CSV for Classical ML 
os.makedirs("data/raw", exist_ok=True)
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
train_df.to_csv("data/raw/train.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)

# 2. Tokenization for DistilBERT [cite: 1140]
print("Tokenizing for DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    # Truncation=True ensures inputs fit model max length [cite: 1144]
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save processed data to disk for the training script
os.makedirs("data/processed", exist_ok=True)
tokenized_datasets.save_to_disk("data/processed/imdb_tokenized")
print("Preprocessing complete.")