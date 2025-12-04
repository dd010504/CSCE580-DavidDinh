import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load Raw Data
print("Loading raw CSV data...")
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

# 2. Vectorization (TF-IDF) 
print("Vectorizing data...")
vectorizer = TfidfVectorizer(max_features=5000) # Limit features for speed
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']

# 3. Train Logistic Regression 
print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 4. Save Model
os.makedirs("models/classical", exist_ok=True)
with open("models/classical/tfidf_logreg.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

# 5. Quick Evaluation
preds = clf.predict(X_test)
print(classification_report(y_test, preds))