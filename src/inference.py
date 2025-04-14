import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_new_data(df):
    # Drop difficulty if present
    df = df.drop(columns=['difficulty'], errors='ignore')

    # Encode categorical columns
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Drop label column if present
    df = df.drop(columns=["label"], errors='ignore')

    # Normalize
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

def run_inference(input_csv, output_csv):
    # Load the model
    model = joblib.load("models/random_forest_model.pkl")

    # Load and preprocess new data
    df = pd.read_csv(input_csv)
    preprocessed_df = preprocess_new_data(df)

    # Predict
    predictions = model.predict(preprocessed_df)

    # Add predictions to original dataframe
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].map({0: "normal", 1: "attack"})

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Predictions saved to: {output_csv}")

if __name__ == "__main__":
    # Example usage
    run_inference("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/sample_input.csv", "/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/sample_predictions.csv")
