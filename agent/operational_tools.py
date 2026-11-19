
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "operational"
    / "bank_transactions_enriched_demo.csv"
)


def get_transaction_details(transaction_id: str) -> dict:
    """Retrieve a transaction using its transaction ID."""

    transaction_id = transaction_id.strip().upper()

    if not transaction_id:
        return {"error": "Please provide a transaction ID."}

    if not DATA_PATH.exists():
        return {"error": "Transaction dataset was not found."}

    df = pd.read_csv(DATA_PATH, dtype={"TransactionID": str})

    matches = df[df["TransactionID"] == transaction_id]

    if matches.empty:
        return {
            "found": False,
            "message": f"No transaction found for {transaction_id}."
        }

    row = matches.iloc[0]

    return {
        "found": True,
        "data_source": "public_sample_dataset",
        "status_source": row["StatusSource"],
        "transaction": {
            "transaction_id": str(row["TransactionID"]),
            "account_id": str(row["AccountID"]),
            "amount": float(row["TransactionAmount"]),
            "date": str(row["TransactionDate"]),
            "type": str(row["TransactionType"]),
            "channel": str(row["Channel"]),
            "location": str(row["Location"]),
            "account_balance": float(row["AccountBalance"]),
            "status": str(row["TransactionStatus"])
        },
        "disclaimer": (
            "Public sample transaction data. "
            "Transaction status is synthetic demo data, "
            "not a verified banking outcome."
        )
    }
