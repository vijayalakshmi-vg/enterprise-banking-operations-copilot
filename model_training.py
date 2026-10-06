from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_preparation import train_loader, label_encoder, tokenizer
from torch.optim import AdamW

MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load DistilBERT for classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

print("Tokenizer loaded")
print("Model loaded")
print("Number of labels:", model.config.num_labels)

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

# Save trained model
model.save_pretrained("./models/distilbert_banking77")
tokenizer.save_pretrained("./models/distilbert_banking77")