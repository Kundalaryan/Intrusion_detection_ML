import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="🛡️ Intrusion Detection", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("/Users/aryankundal/Intrusion_detection_ML/src/Models/random_forest_model.pkl")  # or xgboost_model.pkl

def preprocess_input(df):
    df = df.drop(columns=["difficulty"], errors='ignore')

    expected_columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    # Encode categorical values if present
    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Fill missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # or use np.nan and fillna(mean) if you prefer

    df = df[expected_columns]  # reorder to match training

    # Normalize
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def show_pie_chart(df):
    count = df["prediction_label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(count, labels=count.index, autopct='%1.1f%%', startangle=90, colors=["#66c2a5", "#fc8d62"])
    ax.axis("equal")
    st.subheader("📊 Prediction Distribution")
    st.pyplot(fig)

def compare_with_truth(df):
    if "label" in df.columns:
        label_map = {"normal": 0, "attack": 1}
        df["true_label"] = df["label"].map(label_map)
        df["correct"] = df["true_label"] == df["prediction"]
        accuracy = (df["correct"].sum() / len(df)) * 100
        st.info(f"✅ Accuracy on Uploaded Data: `{accuracy:.2f}%`")
        st.write(df[["prediction", "prediction_label", "label", "correct"]].head())
    else:
        st.write(df[["prediction", "prediction_label"]].head())

# UI
st.title("🛡️ Intrusion Detection System")
st.markdown("Upload a CSV (like `badass_test_data.csv`) to test your model.")

uploaded_file = st.file_uploader("📂 Upload Your Test File", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:", input_df.head())

    try:
        model = load_model()
        processed_df = preprocess_input(input_df.copy())
        predictions = model.predict(processed_df)

        input_df["prediction"] = predictions
        input_df["prediction_label"] = input_df["prediction"].map({0: "normal", 1: "attack"})

        st.success("🎯 Prediction Complete!")
        compare_with_truth(input_df)
        show_pie_chart(input_df)

        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction Results", data=csv, file_name="predicted_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
