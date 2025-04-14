import pandas as pd
import os

# Define column names (41 features + label + difficulty)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

def load_nsl_kdd_data(train_path, test_path):
    """
    Load the NSL-KDD training and testing data.

    Returns:
        train_df (DataFrame): Training data
        test_df (DataFrame): Testing data
    """
    print("[INFO] Loading training and testing data...")
    train_df = pd.read_csv(train_path, names=columns)
    test_df = pd.read_csv(test_path, names=columns)
    print(f"[INFO] Training data shape: {train_df.shape}")
    print(f"[INFO] Testing data shape: {test_df.shape}")
    return train_df, test_df

if __name__ == "__main__":
    # Paths to your local raw files
    train_file = "/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/KDDTrain+.txt"
    test_file = "/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/KDDTest+.txt"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("[ERROR] NSL-KDD dataset files not found. Please download them and place in 'data/raw/'.")
    else:
        train_data, test_data = load_nsl_kdd_data(train_file, test_file)

        # Optional: save as CSV for easier access later
        train_data.to_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/train.csv", index=False)
        test_data.to_csv("/Users/aryankundal/Downloads/intrusion_detection_ml_full/data/raw/test.csv", index=False)
        print("[INFO] Saved as CSV for further processing.")
