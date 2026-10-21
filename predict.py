import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)

MODEL_PATH = os.path.join(
    "models",
    "wildfire_model.keras",
)

CLASS_MAPPING_PATH = os.path.join(
    "models",
    "class_indices.json",
)


# --------------------------------------------------
# Load Class Mapping
# --------------------------------------------------

def load_class_mapping():
    """Load the class mapping generated during training."""

    if not os.path.exists(CLASS_MAPPING_PATH):
        raise FileNotFoundError(
            f"Class mapping not found at {CLASS_MAPPING_PATH}. "
            "Run train_model.py first."
        )

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_indices = json.load(file)

    # Convert {"fire": 0, "nofire": 1}
    # into {0: "fire", 1: "nofire"}
    return {
        int(index): class_name
        for class_name, index in class_indices.items()
    }


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image, image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_wildfire(
    model,
    image_array,
    class_mapping,
):
    """Predict wildfire class using the saved class mapping."""

    score = float(
        model.predict(
            image_array,
            verbose=0,
        )[0][0]
    )

    predicted_index = 1 if score >= 0.5 else 0

    predicted_class = class_mapping[
        predicted_index
    ]

    confidence = (
        score
        if predicted_index == 1
        else 1.0 - score
    )

    # Make class names presentation-friendly
    predicted_class = (
        predicted_class
        .replace("_", " ")
        .title()
    )

    return predicted_class, score, confidence


# --------------------------------------------------
# Display Result
# --------------------------------------------------

def display_result(
    image,
    label,
    score,
    confidence,
):
    """Display wildfire classification results."""

    print("\nWildfire Detection Result")
    print("-" * 30)
    print(f"Predicted Class : {label}")
    print(f"Model Score     : {score:.4f}")
    print(
        f"Confidence      : "
        f"{confidence * 100:.2f}%"
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(
        f"{label} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict wildfire presence "
            "from an input image."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the image to classify",
    )

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_model.py first."
        )

    if not os.path.exists(args.image):
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    print("Loading wildfire detection model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    class_mapping = load_class_mapping()

    print(
        f"Class mapping: {class_mapping}"
    )

    image, image_array = preprocess_image(
        args.image
    )

    label, score, confidence = (
        predict_wildfire(
            model,
            image_array,
            class_mapping,
        )
    )

    display_result(
        image,
        label,
        score,
        confidence,
    )


if __name__ == "__main__":
    main()
