import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import pickle
import pandas as pd

# --- 1. SETUP THE TEST CASES ---
test_cases = [
    # Case 1: Simple
    "I absolutely loved this movie. The acting was fantastic and the plot was exciting.",
    
    # Case 2: Medium
    "The special effects were a bit outdated and the pacing was slow, but the ending was so emotional that it made up for everything. A good watch.",
    
    # Case 3: Complex (Sarcasm/Confusing)
    "I cannot recommend this movie enough to anyone who enjoys being bored to tears for three hours. It was a masterpiece of bad writing."
]

print("Loading models... this might take a minute.")

# --- 2. LOAD FINE-TUNED DISTILBERT ---
ft_path = "./models/distilbert_finetuned"
try:
    tokenizer = DistilBertTokenizer.from_pretrained(ft_path)
    ft_model = DistilBertForSequenceClassification.from_pretrained(ft_path)
    ft_pipeline = pipeline("text-classification", model=ft_model, tokenizer=tokenizer)
    print("[x] Fine-tuned DistilBERT loaded")
except:
    print("[] Could not load Fine-tuned model. Check path.")
    ft_pipeline = None

# --- 3. LOAD BASE DISTILBERT (The "Dumb" Version) ---
base_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print("[x] Base DistilBERT loaded")

# --- 4. LOAD CLASSICAL MODEL (TF-IDF + Logistic Regression) ---
try:
    with open('./models/classical/tfidf_logreg.pkl', 'rb') as f:
        classical_model_bundle = pickle.load(f)
    # Assuming the bundle is a tuple/list: (vectorizer, classifier)
    # Adjust indices if your save format is different
    vectorizer = classical_model_bundle[0] 
    classifier = classical_model_bundle[1]
    print("[x] Classical Model loaded")
except:
    print("[] Could not load Classical model. Check path.")
    classifier = None

# --- 5. LOAD GPT-2 (For comparison as requested by rubric) ---
# Note: GPT-2 is a generator, not a classifier. We check if it generates positive text.
gpt_generator = pipeline("text-generation", model="gpt2")
print("[x] GPT-2 loaded")

# --- 6. RUN PREDICTIONS ---
print("\n--- RESULTS ---\n")

for i, text in enumerate(test_cases):
    print(f"TEST CASE {i+1}: '{text}'")
    
    # 1. Classical Prediction
    if classifier:
        vec_text = vectorizer.transform([text])
        class_pred = classifier.predict(vec_text)[0]
        print(f"   - Classical Model: {class_pred}")
    
    # 2. Base DistilBERT Prediction
    base_pred = base_pipeline(text)[0]
    print(f"   - Base DistilBERT: {base_pred['label']} ({base_pred['score']:.4f})")
    
    # 3. Fine-tuned Prediction
    if ft_pipeline:
        ft_pred = ft_pipeline(text)[0]
        # Map LABEL_0/LABEL_1 to Negative/Positive if needed
        label = "POSITIVE" if ft_pred['label'] == "LABEL_1" else "NEGATIVE" 
        if ft_pred['label'] in ["POSITIVE", "NEGATIVE"]: label = ft_pred['label']
        print(f"   - Fine-tuned BERT: {label} ({ft_pred['score']:.4f})")

    # 4. GPT-2 Generation (Visual check)
    # We ask GPT to complete the sentence to see if it follows the vibe
    gpt_out = gpt_generator(text, max_length=len(text.split())+10, num_return_sequences=1)[0]['generated_text']
    print(f"   - GPT-2 Reaction:  {gpt_out[len(text):]}...") 
    
    print("-" * 50)