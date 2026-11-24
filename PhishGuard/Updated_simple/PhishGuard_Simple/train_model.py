"""Train the PhishGuard phishing URL detector."""

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from app import extract_features, get_group_domain, get_url_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "phishing_dataset.csv"
MODEL_PATH = BASE_DIR / "phishing_model.pkl"


def main():
    print("Loading dataset...")
    if not DATASET_PATH.exists():
        raise FileNotFoundError("phishing_dataset.csv was not found.")

    data = pd.read_csv(DATASET_PATH, usecols=["url", "label"]).dropna()
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["url"] != ""]

    label_numbers = {"legitimate": 0, "phishing": 1}
    data = data[data["label"].isin(label_numbers)]

    urls = data["url"].tolist()
    labels = data["label"].map(label_numbers).tolist()
    if len(set(labels)) != 2:
        raise ValueError("The dataset must contain both legitimate and phishing URLs.")

    print("Total URLs:", len(urls))

    print("Finding registered domains for the train and test split...")
    domains = []
    for url in urls:
        domains.append(get_group_domain(url))

    # GroupShuffleSplit keeps a registered domain out of either the train or test set.
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_rows, test_rows = next(splitter.split(urls, labels, domains))

    train_urls = []
    test_urls = []
    train_labels = []
    test_labels = []

    for row in train_rows:
        train_urls.append(urls[row])
        train_labels.append(labels[row])

    for row in test_rows:
        test_urls.append(urls[row])
        test_labels.append(labels[row])

    print("Training URLs:", len(train_urls))
    print("Testing URLs:", len(test_urls))

    print("Extracting numeric URL features...")
    # Convert each feature dictionary into columns of numbers.
    numeric_vectorizer = DictVectorizer(sparse=True)
    train_features = []
    test_features = []

    for url in train_urls:
        train_features.append(extract_features(url))

    for url in test_urls:
        test_features.append(extract_features(url))

    train_numbers = numeric_vectorizer.fit_transform(train_features)
    test_numbers = numeric_vectorizer.transform(test_features)

    print("Reading URL character patterns...")
    # Convert groups of URL characters into numbers the model can learn from.
    url_vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        n_features=2 ** 18,
        alternate_sign=False
    )

    train_text_values = []
    test_text_values = []
    for url in train_urls:
        train_text_values.append(get_url_text(url))
    for url in test_urls:
        test_text_values.append(get_url_text(url))

    train_text = url_vectorizer.transform(train_text_values)
    test_text = url_vectorizer.transform(test_text_values)

    # Join the numeric features and character patterns into one model input.
    train_matrix = hstack([train_numbers, train_text], format="csr")
    test_matrix = hstack([test_numbers, test_text], format="csr")

    print("Training the model...")
    # Train the classifier using both kinds of URL information.
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.00001,
        max_iter=100,
        tol=0.1,
        average=True,
        random_state=42
    )
    model.fit(train_matrix, train_labels)

    predictions = model.predict(test_matrix)
    print("\nModel results")
    print("Accuracy:", round(accuracy_score(test_labels, predictions) * 100, 2), "%")
    print(confusion_matrix(test_labels, predictions))
    print(classification_report(
        test_labels,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    # Confirm that the two groups do not share any registered domains.
    train_domains = set()
    test_domains = set()
    for row in train_rows:
        train_domains.add(domains[row])
    for row in test_rows:
        test_domains.add(domains[row])
    print("Domains shared by training and testing:", len(train_domains & test_domains))

    joblib.dump({
        "model": model,
        "numeric_vectorizer": numeric_vectorizer,
        "url_vectorizer": url_vectorizer
    }, MODEL_PATH, compress=3)

    print("\nModel saved as phishing_model.pkl")


if __name__ == "__main__":
    main()
