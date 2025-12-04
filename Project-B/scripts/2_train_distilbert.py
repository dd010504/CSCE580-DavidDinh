import json
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load Data
dataset = load_from_disk("data/processed/imdb_tokenized")
# Use a smaller subset for testing if your GPU is slow (optional) [cite: 1017]
# train_dataset = dataset['train'].shuffle(seed=42).select(range(2000)) 
# eval_dataset = dataset['test'].shuffle(seed=42).select(range(500))
train_dataset = dataset['train']
eval_dataset = dataset['test']

# 2. Define Metrics [cite: 1246]
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 3. Initialize Model [cite: 1240]
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 4. Training Arguments [cite: 1258]
training_args = TrainingArguments(
    output_dir="./models/distilbert_finetuned",
    num_train_epochs=3,              # Standard for fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
   eval_strategy="epoch",     # Evaluate every epoch to get loss curves [cite: 890]
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./results/logs',
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# 6. Save Model & History (For Loss Curves)
trainer.save_model("./models/distilbert_finetuned")
with open("./results/metrics/training_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)
print("Training complete.")