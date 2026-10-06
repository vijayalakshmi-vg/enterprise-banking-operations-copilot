from sklearn.metrics import classification_report
import pandas as pd

MODEL_PATH = "./models/distilbert_banking77"


def analyze_errors(
    all_labels,
    all_predictions,
    label_encoder,
    test_df
):

    # Generate classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Remove overall metrics
    class_report = report_df[
        ~report_df.index.isin(
            ["accuracy", "macro avg", "weighted avg"]
        )
    ]

    # Sort classes by recall
    weak_classes = class_report.sort_values(
        "recall",
        ascending=True
    )

    print("\nClasses with lowest recall:\n")

    print(
        weak_classes[
            ["precision", "recall", "f1-score", "support"]
        ].head(10)
    )