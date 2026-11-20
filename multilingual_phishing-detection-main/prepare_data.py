import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split

# 1. Load dataset (change filename to match your actual CSV)
df = pd.read_csv("multilingual_email_phishing_dataset_5000_tamil_malayalam_telugu.csv")

print("Dataset shape:", df.shape)
print(df['label'].value_counts())
print(df['language'].value_counts())
print("Nulls:\n", df.isnull().sum())

# 2. Stratified Train / Val / Test split (70/15/15)
df['strat_key'] = df['label'] + "_" + df['language']

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['strat_key'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['strat_key'], random_state=42
)

for d in [train_df, val_df, test_df]:
    d.drop(columns=['strat_key'], inplace=True)

print("\nTrain:", train_df.shape)
print("Val:  ", val_df.shape)
print("Test: ", test_df.shape)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# 3. NLP Preprocessing
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'\S+@\S+', ' <EMAIL> ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\u0B80-\u0BFF\u0D00-\u0D7F\u0C00-\u0C7Fa-zA-Z0-9<>\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_df(d):
    d = d.copy()
    d['clean_subject'] = d['email_subject'].apply(clean_text)
    d['clean_body'] = d['email_body'].apply(clean_text)
    d['full_text'] = d['clean_subject'] + ' ' + d['clean_body']
    d['text_length'] = d['full_text'].apply(len)
    d['word_count'] = d['full_text'].apply(lambda x: len(x.split()))
    return d

for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
    processed = preprocess_df(split)
    processed.to_csv(f"{name}_preprocessed.csv", index=False)
    print(f"\n{name}_preprocessed.csv saved:", processed.shape)

print("\nDone! Sample full_text:")
print(preprocess_df(train_df)['full_text'].iloc[0][:150])