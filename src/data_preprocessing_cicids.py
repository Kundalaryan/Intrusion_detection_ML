import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def preprocess_cicids(file_path, save_path):
    print(f"[INFO] Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Drop completely empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Remove non-numeric/metadata columns if present
    metadata_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    for col in metadata_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Drop rows with nulls
    df.dropna(inplace=True)

    # Convert 'Label' column to binary (0 = benign, 1 = attack)
    if "Label" in df.columns:
        df = df[df['Label'] != 'Label']  # remove any header rows mistakenly read
        df["Label"] = df["Label"].apply(lambda x: 0 if str(x).lower() == "benign" else 1)
    else:
        print("[ERROR] 'Label' column not found.")
        return

    # Encode 'Protocol' if present
    if "Protocol" in df.columns:
        df["Protocol"] = LabelEncoder().fit_transform(df["Protocol"])

    # Separate features & labels
    features = df.drop(columns=["Label"])
    labels = df["Label"]

    # Normalize numeric features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Add label column back
    features_df["label"] = labels.values

    # Save the output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features_df.to_csv(save_path, index=False)
    print(f"[✅] Preprocessed file saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    input_file = "/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    output_file = "/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed"
    preprocess_cicids(input_file, output_file)
