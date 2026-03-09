import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input


CLASS_NAMES = [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Thickening",
    "Pneumonia",
    "Fibrosis",
    "Edema",
    "Consolidation",
]

IMG_SIZE = (224, 224)
PRIMARY_MODEL = os.path.join("models", "Dense_net_b1_trained_weights.h5")


@st.cache_resource
def load_model(model_path: str, load_mode: str) -> Tuple[tf.keras.Model, str]:
    """Load the selected model with the chosen mode."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    full_model_error = None
    weights_error = None

    if load_mode in ("auto", "full"):
        try:
            full_model = tf.keras.models.load_model(model_path, compile=False)
            return full_model, "full model"
        except Exception as e:
            full_model_error = e
            if load_mode == "full":
                raise

    # Match training architecture:
    # Sequential([DenseNet121(include_top=False), GAP, Dense(14, sigmoid)])
    base = tf.keras.applications.DenseNet121(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights=None,
        include_top=False,
    )
    base.trainable = True
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="sigmoid"),
        ]
    )

    try:
        model.load_weights(model_path)
        return model, "weights"
    except Exception as e:
        weights_error = e
        if load_mode == "weights":
            raise

    raise RuntimeError(
        "Could not load model as full model or weights.\n"
        f"Full model error: {full_model_error}\n"
        f"Weights error: {weights_error}"
    )


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """Convert uploaded image to model-ready tensor."""

    img = uploaded_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def main() -> None:
    st.set_page_config(page_title="Chest X-ray Multi-label Prediction", layout="centered")

    st.title("Chest X-ray Disease Prediction (DenseNet121)")
    st.write("Upload a chest X-ray image to get multi-label predictions for 14 classes.")

    model_files = (
        sorted(
            [
                os.path.join("models", f)
                for f in os.listdir("models")
                if f.lower().endswith(".h5")
            ]
        )
        if os.path.isdir("models")
        else []
    )

    if not model_files:
        st.error("No `.h5` model files found in `models/`.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        default_index = model_files.index(PRIMARY_MODEL) if PRIMARY_MODEL in model_files else 0
        selected_model = st.selectbox("Select model file", model_files, index=default_index)
        load_mode_label = st.selectbox(
            "Model load mode",
            ["Auto (Recommended)", "Weights only", "Full model"],
            index=1,
        )
        load_mode = {
            "Auto (Recommended)": "auto",
            "Weights only": "weights",
            "Full model": "full",
        }[load_mode_label]
        threshold = st.slider("Prediction threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

    try:
        model, load_kind = load_model(selected_model, load_mode)
        st.success(f"Model loaded from: `{selected_model}` ({load_kind})")
    except Exception as e:
        st.error("Failed to load model. Check architecture/weights compatibility.")
        st.exception(e)
        st.stop()

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Please upload an image file to run prediction.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("Predict", type="primary"):
        input_tensor = preprocess_image(image)
        probs = model.predict(input_tensor, verbose=0)[0]

        results_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
        results_df = results_df.sort_values("Probability", ascending=False).reset_index(drop=True)

        st.subheader("Predictions")
        st.dataframe(
            results_df.style.format({"Probability": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        predicted = results_df[results_df["Probability"] >= threshold]

        st.subheader(f"Classes above threshold ({threshold:.2f})")
        if predicted.empty:
            st.write("No class is above the selected threshold.")
        else:
            st.write(
                ", ".join(
                    [f"{row.Class} ({row.Probability:.3f})" for row in predicted.itertuples(index=False)]
                )
            )


if __name__ == "__main__":
    main()
