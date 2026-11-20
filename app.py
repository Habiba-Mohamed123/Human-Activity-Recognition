
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import os

# -----------------------------
# 1. Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Human Activity Recognition")
st.write(
    "AI-powered system for recognizing human activities "
    "using smartphone sensor features."
)

# -----------------------------
# 2. Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# 3. Model Architecture
# IMPORTANT:
# This architecture must exactly match
# AdvancedHARWithoutAttention used in Kaggle.
# -----------------------------

class AdvancedHARWithoutAttention(nn.Module):

    def __init__(
        self,
        input_dim=561,
        num_classes=6,
        hidden_dim=256,
        dropout=0.3
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.activation = nn.GELU()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.input_layer(x)

        residual = x

        x = self.residual_block(x)

        x = self.activation(x + residual)

        output = self.classifier(x)

        return output


# -----------------------------
# 4. Load Saved Files
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "HAR_Demo")

@st.cache_resource
def load_artifacts():

    with open(
        os.path.join(MODEL_DIR, "metadata.json"),
        "r"
    ) as f:
        metadata = json.load(f)

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    model = AdvancedHARWithoutAttention(
        input_dim=metadata["input_dim"],
        num_classes=metadata["num_classes"],
        hidden_dim=metadata["hidden_dim"],
        dropout=metadata["dropout"]
    )

    model_path = os.path.join(
        MODEL_DIR,
        "har_model.pth"
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, scaler, label_encoder, metadata


# -----------------------------
# 5. Upload Dataset
# -----------------------------

st.subheader("📂 Upload Sensor Features")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing 561 features",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        model, scaler, encoder, metadata = load_artifacts()

        feature_columns = metadata["feature_columns"]

        # Accept the same 561 features used during training.
        missing_columns = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing {len(missing_columns)} required features."
            )

        elif len(df) == 0:
            st.error("The uploaded CSV is empty.")

        else:
            X = df[feature_columns].copy()

            if not np.isfinite(X.to_numpy(dtype=float)).all():
                st.error("The file contains missing or infinite values.")

            else:
                X_scaled = scaler.transform(X)

                X_tensor = torch.tensor(
                    X_scaled,
                    dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    logits = model(X_tensor)
                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    ).cpu().numpy()

                predictions = np.argmax(
                    probabilities,
                    axis=1
                )

                labels = encoder.inverse_transform(predictions)

                st.success("Prediction completed!")

                results = pd.DataFrame({
                    "Predicted Activity": labels,
                    "Confidence": probabilities.max(axis=1) * 100
                })

                st.subheader("🎯 Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.subheader("📊 Activity Probabilities")

                row_index = st.selectbox(
                    "Select sample",
                    range(len(df))
                )

                chart_data = pd.DataFrame({
                    "Activity": encoder.classes_,
                    "Probability": probabilities[row_index]
                }).set_index("Activity")

                st.bar_chart(chart_data)

                st.metric(
                    "Predicted Activity",
                    labels[row_index]
                )

                st.metric(
                    "Confidence",
                    f"{probabilities[row_index].max() * 100:.2f}%"
                )

    except Exception as e:
        st.error(f"Error: {e}")





