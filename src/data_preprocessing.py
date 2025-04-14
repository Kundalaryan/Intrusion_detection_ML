# File: src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_data():
    train = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/train.csv")
    test = pd.read_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/test.csv")
    return train, test

def drop_unused(df):
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
    return df

def encode_categorical(df, encoders=None):
    cat_cols = ["protocol_type", "service", "flag"]
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])
    return df, encoders

def encode_labels(df):
    df['label'] = df['label'].apply(lambda x: 0 if x == "normal" else 1)
    return df

def normalize(df, scaler=None):
    feature_cols = df.columns.drop('label')
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler

def preprocess_data():
    train, test = load_data()

    train = drop_unused(train)
    test = drop_unused(test)

    train, encoders = encode_categorical(train)
    test, _ = encode_categorical(test, encoders)

    train = encode_labels(train)
    test = encode_labels(test)

    train, scaler = normalize(train)
    test, _ = normalize(test, scaler)

    train.to_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/train_preprocessed.csv", index=False)
    test.to_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/processed/test_preprocessed.csv", index=False)
    print("[INFO] Preprocessed files saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()
