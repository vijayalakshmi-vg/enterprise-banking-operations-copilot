
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

transactions = pd.read_csv(DATA_DIR / "bank_transactions.csv")

# Verify transaction IDs are suitable for a one-to-one merge
assert transactions["TransactionID"].notna().all()
assert transactions["TransactionID"].is_unique

# Reproducible synthetic statuses — NOT real bank outcomes
rng = np.random.default_rng(42)

statuses = pd.DataFrame({
    "TransactionID": transactions["TransactionID"],
    "TransactionStatus": rng.choice(
        ["SUCCESS", "PENDING", "FAILED"],
        size=len(transactions),
        p=[0.85, 0.05, 0.10]
    ),
    "StatusSource": "synthetic_demo"
})

statuses.to_csv(DATA_DIR / "transaction_status_demo.csv", index=False)

merged = transactions.merge(
    statuses,
    on="TransactionID",
    how="left",
    validate="one_to_one"
)

merged.to_csv(
    DATA_DIR / "bank_transactions_enriched_demo.csv",
    index=False
)

print("Original records:", len(transactions))
print("Merged records:", len(merged))
print("\nSynthetic status distribution:")
print(merged["TransactionStatus"].value_counts())
print("\nFiles created successfully.")
