from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv")
X = df.drop(columns=["label"])
y = df["label"]

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid.fit(X, y)

print("[INFO] Best Parameters:", grid.best_params_)
print("[INFO] Accuracy:", grid.best_score_)

joblib.dump(grid.best_estimator_, "models/random_forest_tuned.pkl")
