import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

test_df = pd.read_csv("test_preprocessed.csv")
label_map = {"Phishing": 1, "Legitimate": 0}
y_true = test_df['label'].map(label_map).values

X_test = np.load("X_test.npy")
y_test_lstm = np.load("y_test.npy")

results = {}

def get_model_size(path):
    total = 0
    if os.path.isdir(path):
        for f in os.listdir(path):
            total += os.path.getsize(os.path.join(path, f))
    else:
        total = os.path.getsize(path)
    return round(total / (1024 * 1024), 2)  # MB

# ---- LSTM ----
print("Evaluating LSTM...")
lstm_model = tf.keras.models.load_model("lstm_model.keras")
start = time.time()
preds = (lstm_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
infer_time = round((time.time() - start) / len(X_test) * 1000, 3)  # ms/sample
results["LSTM"] = {
    "accuracy": (preds == y_test_lstm).mean(),
    "precision": precision_score(y_test_lstm, preds),
    "recall": recall_score(y_test_lstm, preds),
    "f1": f1_score(y_test_lstm, preds),
    "size_MB": get_model_size("lstm_model.keras"),
    "infer_ms_per_sample": infer_time
}

# ---- BiLSTM ----
print("Evaluating BiLSTM...")
bilstm_model = tf.keras.models.load_model("bilstm_model.keras")
start = time.time()
preds = (bilstm_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
infer_time = round((time.time() - start) / len(X_test) * 1000, 3)
results["BiLSTM"] = {
    "accuracy": (preds == y_test_lstm).mean(),
    "precision": precision_score(y_test_lstm, preds),
    "recall": recall_score(y_test_lstm, preds),
    "f1": f1_score(y_test_lstm, preds),
    "size_MB": get_model_size("bilstm_model.keras"),
    "infer_ms_per_sample": infer_time
}

# ---- mBERT / XLM-R helper ----
def evaluate_transformer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    preds = []
    start = time.time()
    with torch.no_grad():
        for text in test_df['full_text'].astype(str).tolist():
            enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            out = model(**enc)
            pred = torch.argmax(out.logits, dim=1).item()
            preds.append(pred)
    infer_time = round((time.time() - start) / len(preds) * 1000, 3)
    preds = np.array(preds)
    size = 0
    for f in os.listdir(model_path):
        size += os.path.getsize(os.path.join(model_path, f))
    return {
        "accuracy": (preds == y_true).mean(),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "size_MB": round(size / (1024*1024), 2),
        "infer_ms_per_sample": infer_time
    }

print("Evaluating mBERT...")
results["mBERT"] = evaluate_transformer("mbert_model")

print("Evaluating XLM-R...")
results["XLM-R"] = evaluate_transformer("xlmr_model")

# ---- Final comparison table ----
df = pd.DataFrame(results).T
df = df.round(4)
print("\n=== Model Comparison ===")
print(df)

df.to_csv("model_comparison.csv")
print("\nSaved as model_comparison.csv")

best_model = df['f1'].idxmax()
print(f"\nBest model (by F1-score): {best_model}")