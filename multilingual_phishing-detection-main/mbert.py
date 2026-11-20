import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# Load preprocessed data
train_df = pd.read_csv("train_preprocessed.csv")
val_df = pd.read_csv("val_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")

label_map = {"Phishing": 1, "Legitimate": 0}
train_df['label_id'] = train_df['label'].map(label_map)
val_df['label_id'] = val_df['label'].map(label_map)
test_df['label_id'] = test_df['label'].map(label_map)

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128

print("Loading mBERT tokenizer... (downloads once, then cached)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = PhishingDataset(train_df['full_text'].tolist(), train_df['label_id'].tolist(), tokenizer, MAX_LEN)
val_dataset = PhishingDataset(val_df['full_text'].tolist(), val_df['label_id'].tolist(), tokenizer, MAX_LEN)
test_dataset = PhishingDataset(test_df['full_text'].tolist(), test_df['label_id'].tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

print("Loading mBERT model... (downloads once, then cached)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3  # transformers need very few epochs to fine-tune

def evaluate(loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    return accuracy_score(actuals, preds)

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1} done. Avg Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

test_acc = evaluate(test_loader)
print(f"\nmBERT Test Accuracy: {test_acc:.4f}")

model.save_pretrained("mbert_model")
tokenizer.save_pretrained("mbert_model")
print("\nModel saved to mbert_model/ folder")