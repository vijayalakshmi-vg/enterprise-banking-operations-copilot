
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Locate saved model artifacts
MODEL_DIR = Path("models/distilbert_banking77")

# 2. Load the saved tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 3. Load the original intent names
label_classes = np.load(
    MODEL_DIR / "label_classes.npy",
    allow_pickle=True
)
# 4. Use evaluation mode for inference
model.eval()


def predict_intent(text):
    # Convert text into model inputs
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Find the class with the highest logit
    predicted_id = torch.argmax(outputs.logits, dim=1).item()

    # Convert numeric ID back to the original intent name
    predicted_intent = label_classes[predicted_id]

    # Calculate confidence from the class probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence = probabilities[0, predicted_id].item()

    return predicted_intent, confidence



if __name__ == "__main__":
    test_questions = [
        "I lost my bank card",
        "Why was my card declined?",
        "My ATM withdrawal is still pending",
        "Do I pay fees when using my card abroad?",
        "I see a transaction I don't recognize",
    ]

    for question in test_questions:
        intent, confidence = predict_intent(question)

        print(f"\nQuestion: {question}")
        print(f"Predicted intent: {intent}")
        print(f"Confidence: {confidence:.2%}")

