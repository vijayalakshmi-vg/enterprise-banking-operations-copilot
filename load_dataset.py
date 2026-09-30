import pandas as pd

train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns)

print("\nFirst 5 rows:")
print(train_df.tail())

print("\n--- Train Info ---")
print(train_df.info())

print("\n--- Test Info ---")
print(test_df.info())

print("\n--- Missing Values ---")
print(train_df.isnull().sum())

print("\n--- Number of Unique Intents ---")
print(train_df["category"].nunique())

print("\n--- Intent Distribution ---")
print(train_df["category"].value_counts())

print("\n--- Sample Records ---")
print(train_df.sample(5, random_state=42))

#preprocessing the text

import re

# Separate features and target
X_train = train_df["text"]
y_train = train_df["category"]

X_test = test_df["text"]
y_test = test_df["category"]


# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)


# Check before and after
print("Before:")
print(X_train.iloc[20])

print("\nAfter:")
print(X_train_clean.iloc[20])

#Tf- IDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_test_tfidf = vectorizer.transform(X_test_clean)

print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)

# Check before and after

print("\before:")
print(X_train_clean.iloc[20])

print("\nTF-IDF vector:")
print(X_train_tfidf[20].toarray())

# Word + TF-IDF value
feature_names = vectorizer.get_feature_names_out()
tfidf_values = X_train_tfidf[0].toarray()[0]

for word, value in zip(feature_names, tfidf_values):
    if value > 0:
        print(word, ":", value)


#Train logistic regression baseline

from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

print("Predictions:", y_pred[:10])

# Evaluation code

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

#inspect actual mistakes rather than just the overall score.

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix Shape:", cm.shape)
print(cm)

#heatmap
import numpy as np

# Remove correct predictions
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

# Get indices sorted by number of mistakes
indices = np.dstack(
    np.unravel_index(np.argsort(cm_no_diag.ravel())[::-1], cm_no_diag.shape)
)[0]

print("Top 15 confusions:\n")

count = 0
for actual_idx, predicted_idx in indices:
    mistakes = cm_no_diag[actual_idx, predicted_idx]

    if mistakes == 0:
        break

    print(
        f"{model.classes_[actual_idx]}"
        f"  →  "
        f"{model.classes_[predicted_idx]}"
        f" : {mistakes}"
    )

    count += 1
    if count == 15:
        break


# stopword experiment

# TF-IDF with stopword removal
vectorizer_stop = TfidfVectorizer(stop_words="english")

X_train_tfidf_stop = vectorizer_stop.fit_transform(X_train_clean)
X_test_tfidf_stop = vectorizer_stop.transform(X_test_clean)

# Train Logistic Regression
model_stop = LogisticRegression(max_iter=1000)
model_stop.fit(X_train_tfidf_stop, y_train)

# Predict
y_pred_stop = model_stop.predict(X_test_tfidf_stop)

# Evaluate
print(classification_report(y_test, y_pred_stop))