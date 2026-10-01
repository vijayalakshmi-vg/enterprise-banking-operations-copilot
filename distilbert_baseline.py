from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load DistilBERT for classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=77
)

print("Tokenizer loaded")
print("Model loaded")
print("Number of labels:", model.config.num_labels)

#load dataset

import pandas as pd

train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

#Now let's tokenize the text.

X_train = train_df["text"].tolist()
X_test = test_df["text"].tolist()

y_train = train_df["category"].tolist()
y_test = test_df["category"].tolist()

# Tokenize
train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization complete")
print("input_ids shape:", len(train_encodings["input_ids"]))
print("First input_ids:", train_encodings["input_ids"][0][:20])
print("First attention_mask:", train_encodings["attention_mask"][0][:20])

for i in range(5):
    print(f"\n--- Example {i+1} ---")
    print("Original:", X_train[i])
    print("Tokens:", tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][i]))
    print("Input IDs:", train_encodings["input_ids"][i])
    print("Attention Mask:", train_encodings["attention_mask"][i])

#convert BANKING77 intent names → numerical labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Number of classes:", len(label_encoder.classes_))
print("First 10 classes:", label_encoder.classes_[:10])
print("First 10 encoded labels:", y_train_encoded[:10])

for i in range(10):
    print(
        i,
        "Text:", X_train[i],
        "| Original:", y_train[i],
        "| Encoded:", y_train_encoded[i]
    )
    
#converting tokenized labels to pytorch datasets
import torch
from torch.utils.data import Dataset

class BankingDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item


train_dataset = BankingDataset(
    train_encodings,
    y_train_encoded
)

test_dataset = BankingDataset(
    test_encodings,
    y_test_encoded
)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

print("\nFirst training example:")
print(train_dataset[0])