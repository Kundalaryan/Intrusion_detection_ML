import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def load_data():
    train = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv")
    test = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/test_preprocessed.csv")

    X_train = train.drop(columns=["label"])
    y_train = train["label"]

    X_test = test.drop(columns=["label"])
    y_test = test["label"]

    return X_train, y_train, X_test, y_test

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\n[INFO] Accuracy: {acc:.4f}")
    print("\n[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, predictions))

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    print("[INFO] Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print("[INFO] Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
