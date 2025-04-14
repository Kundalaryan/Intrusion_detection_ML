import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score

# Load preprocessed dataset
df = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv")
X = df.drop(columns=["label"])
y = df["label"]

# Load model
model = joblib.load("models/xgboost_model.pkl")  # or random_forest_tuned.pkl / xgboost_model.pkl

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Mean Accuracy: %.2f%%" % (scores.mean() * 100))
print("Standard Deviation: %.2f%%" % (scores.std() * 100))
