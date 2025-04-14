import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv")
X = df.drop(columns=["label"])
y = df["label"]

model = XGBClassifier(n_estimators=150, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)

preds = model.predict(X)
print(classification_report(y, preds))

joblib.dump(model, "models/xgboost_model.pkl")
