import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load preprocessed data
train_df = pd.read_csv("train_preprocessed.csv")
val_df = pd.read_csv("val_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")

train_texts = train_df['full_text'].astype(str).tolist()
val_texts = val_df['full_text'].astype(str).tolist()
test_texts = test_df['full_text'].astype(str).tolist()

# 1. Fit tokenizer ONLY on train data (prevents data leakage)
VOCAB_SIZE = 10000
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

print("Vocabulary size found:", len(tokenizer.word_index))

# 2. Convert texts to sequences of numbers
train_seq = tokenizer.texts_to_sequences(train_texts)
val_seq = tokenizer.texts_to_sequences(val_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)

# 3. Pad sequences to a fixed length
MAX_LEN = 100
X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val = pad_sequences(val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# 4. Encode labels (Phishing=1, Legitimate=0)
label_map = {"Phishing": 1, "Legitimate": 0}
y_train = train_df['label'].map(label_map).values
y_val = val_df['label'].map(label_map).values
y_test = test_df['label'].map(label_map).values

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 5. Save everything for next step (LSTM model)
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nDone! Tokenizer and padded sequences saved.")