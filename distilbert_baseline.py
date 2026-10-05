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


#DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("Number of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))

#let's actually look at one batch
batch = next(iter(train_loader))

print("Batch keys:", batch.keys())
print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["labels"].shape)

print("\nLabels in this batch:")
print(batch["labels"])

#send one batch into DistilBERT before training.
# Take one batch
batch = next(iter(train_loader))

# Send the batch to DistilBERT
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"]
)

print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss)

#Before training, let's inspect one sentence's 77 logits and see how the model chooses its current prediction.

print("First example logits:")
print(outputs.logits[0])

predicted_class = torch.argmax(outputs.logits[0]).item()

print("\nPredicted class:", predicted_class)
print("Actual class:", batch["labels"][0])
print("Predicted intent:", label_encoder.inverse_transform([predicted_class])[0])
print("Actual intent:", label_encoder.inverse_transform([batch["labels"][0].item()])[0])


#train one batch using optimiser
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=5e-5
)
#check for 2 epochs
num_epochs = 2

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
from sklearn.metrics import accuracy_score, classification_report

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)

print("DistilBERT Accuracy:", accuracy)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=label_encoder.classes_
    )
)