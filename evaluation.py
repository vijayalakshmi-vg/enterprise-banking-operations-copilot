import torch
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, test_loader, label_encoder):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("DistilBERT Accuracy:", accuracy)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=label_encoder.classes_
        )
    )

    return all_labels, all_predictions