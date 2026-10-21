import argparse
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


# --------------------------------------------------
# Load and Preprocess Image
# --------------------------------------------------

def preprocess_image(image_path):
    """Load and preprocess an image for wildfire prediction."""

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image, image_array


# --------------------------------------------------
# Predict
# --------------------------------------------------

def predict_wildfire(model, image_array):
    """Predict whether an image contains wildfire."""

    prediction = float(
        model.predict(
            image_array,
            verbose=0,
        )[0][0]
    )

    # The dataset/model convention used by this project
    # treats values below 0.5 as Fire and values at or
    # above 0.5 as No Fire.
    if prediction < 0.5:
        label = "Fire"
        confidence = 1.0 - prediction
    else:
        label = "No Fire"
        confidence = prediction

    return label, prediction, confidence


# --------------------------------------------------
# Display Result
# --------------------------------------------------

def display_result(
    image,
    label,
    prediction,
    confidence,
):
    """Display the image and prediction result."""

    print("\nWildfire Detection Result")
    print("-" * 30)
    print(f"Predicted Class : {label}")
    print(f"Model Score     : {prediction:.4f}")
    print(f"Confidence      : {confidence * 100:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    plt.title(
        f"{label} | Confidence: {confidence * 100:.2f}%"
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
            "Predict wildfire presence from an input image."
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

    image, image_array = preprocess_image(
        args.image
    )

    label, prediction, confidence = (
        predict_wildfire(
            model,
            image_array,
        )
    )

    display_result(
        image,
        label,
        prediction,
        confidence,
    )


if __name__ == "__main__":
    main()
