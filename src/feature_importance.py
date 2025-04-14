import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv")
X = df.drop(columns=["label"])
y = df["label"]

model = joblib.load("models/random_forest_model.pkl")
importances = model.feature_importances_

plt.figure(figsize=(10, 6))
pd.Series(importances, index=X.columns).sort_values().plot(kind='barh')
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
