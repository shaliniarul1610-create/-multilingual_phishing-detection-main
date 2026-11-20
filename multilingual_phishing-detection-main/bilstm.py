import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load tokenized data (from Step 6)
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy").astype("float32")
y_val = np.load("y_val.npy").astype("float32")
y_test = np.load("y_test.npy").astype("float32")

VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128

# BiLSTM model — same as LSTM but wrapped with Bidirectional()
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.build(input_shape=(None, MAX_LEN))
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nBiLSTM Test Accuracy: {test_acc:.4f}")
print(f"BiLSTM Test Loss: {test_loss:.4f}")

model.save("bilstm_model.keras")
print("\nModel saved as bilstm_model.keras")