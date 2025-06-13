import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import re
import time
import joblib

# Helper to load data

def load_data(path):
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'__label__(\S+)\s+(.*)', line)
            if m:
                labels.append(m.group(1))
                texts.append(m.group(2))
    return texts, labels

def main():
    input_path = "/home/user/cs336-a4-data/datagen/train.txt"
    valid_path = "/home/user/cs336-a4-data/datagen/valid.txt"

    X_train, y_train = load_data(input_path)
    X_valid, y_valid = load_data(valid_path)

    wandb.init(project="sklearn-quality-classifier", config={
        "model": "SGDClassifier",
        "vectorizer": "TfidfVectorizer",
        "train_size": len(X_train),
        "valid_size": len(X_valid),
        "epochs": 20,
    })

    # Vectorize data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    # Get all unique classes
    classes = np.unique(y_train)

    # SGDClassifier supports partial_fit for per-epoch training
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)

    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        clf.partial_fit(X_train_vec, y_train, classes=classes)
        epoch_time = time.time() - start_time

        # Training metrics
        y_train_pred = clf.predict(X_train_vec)
        y_train_proba = clf.predict_proba(X_train_vec)
        train_loss = log_loss(y_train, y_train_proba, labels=classes)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Validation metrics
        y_valid_pred = clf.predict(X_valid_vec)
        y_valid_proba = clf.predict_proba(X_valid_vec)
        valid_loss = log_loss(y_valid, y_valid_proba, labels=classes)
        valid_acc = accuracy_score(y_valid, y_valid_pred)

        metrics = {
            "epoch": epoch,
            "time": epoch_time,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "valid/loss": valid_loss,
            "valid/accuracy": valid_acc,
        }
        print(f"Epoch {epoch}: {metrics}")
        wandb.log(metrics)

    # Final metrics
    print("\nFinal Training and Validation Metrics:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")
    wandb.log({
        "final/train_loss": train_loss,
        "final/train_accuracy": train_acc,
        "final/valid_loss": valid_loss,
        "final/valid_accuracy": valid_acc,
    })

    # Save the vectorizer and classifier
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(clf, "classifier.joblib")

    wandb.finish()

if __name__ == "__main__":
    main()