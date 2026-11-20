import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100
test_df = pd.read_csv("test_preprocessed.csv")

label_map = {"Phishing": 1, "Legitimate": 0}
test_df['label_id'] = test_df['label'].map(label_map)

print("=== Per-Language Accuracy (Real, computed from test set) ===\n")

for lang in ["Tamil", "Malayalam", "Telugu"]:
    subset = test_df[test_df['language'] == lang]
    if len(subset) == 0:
        print(f"{lang}: no samples in test set")
        continue

    seq = tokenizer.texts_to_sequences(subset['full_text'].astype(str).tolist())
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    preds = (model.predict(padded, verbose=0) > 0.5).astype(int).flatten()
    actual = subset['label_id'].values

    accuracy = (preds == actual).mean() * 100
    print(f"{lang:12s} → {accuracy:.1f}% Accuracy  ({len(subset)} test samples)")