import torch
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm

# --- PART 1: AI TEST CASES FOR FINE-TUNED MODEL (Missing from your Table) ---
print("\n=== PART 1: Running AI Test Cases on Fine-Tuned Model ===")
test_cases = [
    "I absolutely loved this movie. The acting was fantastic and the plot was exciting.",
    "The special effects were a bit outdated and the pacing was slow, but the ending was so emotional that it made up for everything. A good watch.",
    "I cannot recommend this movie enough to anyone who enjoys being bored to tears for three hours. It was a masterpiece of bad writing."
]

ft_path = "./models/distilbert_finetuned"
device = 0 if torch.cuda.is_available() else -1

try:
    print(f"Loading Fine-Tuned Model from {ft_path}...")
    ft_model = DistilBertForSequenceClassification.from_pretrained(ft_path)
    ft_tokenizer = DistilBertTokenizer.from_pretrained(ft_path)
    ft_pipe = pipeline("text-classification", model=ft_model, tokenizer=ft_tokenizer, device=device)
    
    print("\nRESULTS FOR TABLE (Fine-Tuned DistilBERT):")
    for i, text in enumerate(test_cases):
        result = ft_pipe(text)[0]
        # Map generic labels to Positive/Negative if needed
        label = result['label']
        if label == "LABEL_1": label = "POSITIVE"
        if label == "LABEL_0": label = "NEGATIVE"
        
        print(f"Test Case {i+1}: {label} (Confidence: {result['score']:.4f})")
        
except Exception as e:
    print(f"Could not load Fine-Tuned model: {e}")
    print("Ensure you ran Script 2 and the 'models/distilbert_finetuned' folder exists.")


# --- PART 2: METRICS FOR BASE & GPT-2 (For Performance Table) ---
print("\n=== PART 2: Calculating Missing Metrics (Base & GPT-2) ===")

# 1. Setup - Use a small sample to be fast (e.g., 100 reviews)
print("Loading Test Dataset...")
try:
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(100))
    texts = dataset["text"]
    labels = dataset["label"] # 0 = Neg, 1 = Pos
except:
    print("Could not load IMDB dataset. Skipping Part 2.")
    exit()

# 2. Define the Zero-Shot pipelines
print("Loading Baseline Models...")

# Base DistilBERT (Zero-Shot Pipeline to get better results than raw head)
base_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# GPT-2 (Generative approach)
gpt_pipe = pipeline("text-generation", model="gpt2", device=device)

# 3. Run Predictions
base_preds = []
gpt_preds = []

print(f"Running evaluation on {len(texts)} samples...")

for text in tqdm(texts):
    # --- Base DistilBERT ---
    b_out = base_pipe(text[:512], truncation=True)[0]
    if b_out['label'] in ['POSITIVE', 'LABEL_1']:
        base_preds.append(1)
    else:
        base_preds.append(0)

    # --- GPT-2 ---
    prompt = f"Review: {text[:400]}\nSentiment (Positive or Negative):"
    g_out = gpt_pipe(prompt, max_new_tokens=3, pad_token_id=50256)[0]['generated_text']
    generated = g_out[len(prompt):].lower()
    if "positive" in generated:
        gpt_preds.append(1)
    else:
        gpt_preds.append(0)

# 4. Calculate Metrics
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-Score:  {f1:.3f}")

print_metrics("Base DistilBERT (Zero-Shot)", labels, base_preds)
print_metrics("GPT-2 (Zero-Shot)", labels, gpt_preds)