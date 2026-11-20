import numpy as np
import pandas as pd
import pickle

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

train_df = pd.read_csv("train_preprocessed.csv")

# 1. Check label balance
print("y_train unique values and counts:")
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# 2. Check X_train isn't all zeros/broken
print("\nX_train dtype:", X_train.dtype)
print("X_train sample row (first 20 numbers):", X_train[0][:20])
print("Non-zero tokens in first row:", np.count_nonzero(X_train[0]))

# 3. CRITICAL: decode X_train back to words and compare with original text + label
index_word = tokenizer.index_word
decoded = ' '.join([index_word.get(i, '?') for i in X_train[0] if i != 0])

print("\n--- Alignment check (row 0) ---")
print("Decoded from X_train:", decoded[:150])
print("Original full_text:  ", str(train_df['full_text'].iloc[0])[:150])
print("y_train label:", y_train[0], "| Original label:", train_df['label'].iloc[0])