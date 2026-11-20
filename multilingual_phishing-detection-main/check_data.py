import pandas as pd

train_df = pd.read_csv("train_preprocessed.csv")

# 1. Check if URL content had useful signal
print("--- Sample URLs by label ---")
print("Phishing URLs:\n", train_df[train_df['label']=='Phishing']['url'].head(5).tolist())
print("\nLegitimate URLs:\n", train_df[train_df['label']=='Legitimate']['url'].head(5).tolist())

# 2. Check sender_email patterns
print("\n--- Sample sender_email by label ---")
print("Phishing senders:\n", train_df[train_df['label']=='Phishing']['sender_email'].head(5).tolist())
print("\nLegitimate senders:\n", train_df[train_df['label']=='Legitimate']['sender_email'].head(5).tolist())

# 3. Quick baseline check: does full_text alone separate the classes?
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train_df['full_text'].astype(str)
y = train_df['label'].map({"Phishing": 1, "Legitimate": 0})

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
vec = TfidfVectorizer(max_features=5000)
X_tr_vec = vec.fit_transform(X_tr)
X_te_vec = vec.transform(X_te)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr_vec, y_tr)
preds = clf.predict(X_te_vec)
print("\n--- Baseline TF-IDF + LogisticRegression accuracy ---")
print(accuracy_score(y_te, preds))