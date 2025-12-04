import torch
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, GPT2ForSequenceClassification, GPT2Tokenizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODELS ---
# 1. Fine-Tuned DistilBERT
ft_model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_finetuned").to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 2. Base DistilBERT (Untrained) [cite: 873]
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# 3. GPT-2 (Base) [cite: 876]
# Note: Since GPT-2 is causal, we use SequenceClassification head which is initialized randomly. 
# This serves as a "base" (untrained head) comparison.
gpt_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token # Fix padding for GPT

# 4. Classical Model
with open("models/classical/tfidf_logreg.pkl", "rb") as f:
    vectorizer, clf = pickle.load(f)

# --- HELPER FUNCTIONS ---
def predict_bert(model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=1).item()

def predict_gpt(model, text):
    inputs = gpt_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=1).item()

# --- PART A: AI TEST CASES [cite: 882] ---
print("\n--- AI Test Cases ---")
test_cases = [
    "This movie was an absolute masterpiece with stunning visuals.", # Simple Positive
    "I wasted two hours of my life on this garbage.",               # Simple Negative
    "The acting was okay, but the plot was a disaster."             # Mixed/Complex Negative
]

results = []
for text in test_cases:
    res = {
        "Input": text,
        "Fine-Tuned": predict_bert(ft_model, text),
        "Base DistilBERT": predict_bert(base_model, text),
        "GPT-2": predict_gpt(gpt_model, text),
        "Classical": clf.predict(vectorizer.transform([text]))[0]
    }
    results.append(res)
    
print(pd.DataFrame(results))

# --- PART B: CONFUSION MATRICES [cite: 892] ---
# Load a subset of test data for visualization
test_data = pd.read_csv("data/raw/test.csv").sample(500, random_state=42) # Sample for speed
y_true = test_data['label'].tolist()

# Generate Predictions
y_pred_ft = [predict_bert(ft_model, t) for t in test_data['text']]
y_pred_base = [predict_bert(base_model, t) for t in test_data['text']]
y_pred_classical = clf.predict(vectorizer.transform(test_data['text']))

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"results/figures/{filename}")
    plt.close()

plot_cm(y_true, y_pred_ft, "Fine-Tuned DistilBERT", "cm_finetuned.png")
plot_cm(y_true, y_pred_base, "Base DistilBERT", "cm_base.png")
plot_cm(y_true, y_pred_classical, "Classical ML", "cm_classical.png")
print("\nConfusion matrices saved to results/figures/")

# --- PART C: METRICS COMPARISON TABLE [cite: 899] ---
metrics = []
for name, preds in [("Fine-Tuned", y_pred_ft), ("Base", y_pred_base), ("Classical", y_pred_classical)]:
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, preds),
        "F1 Score": f1_score(y_true, preds)
    })

pd.DataFrame(metrics).to_csv("results/metrics/comparison_table.csv", index=False)
print("Metrics saved to results/metrics/comparison_table.csv")