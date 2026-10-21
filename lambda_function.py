import io
import json
import os

import boto3
import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "wildfire_model.keras",
)

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_indices.json",
)


# --------------------------------------------------
# Load Model and Class Mapping
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)


def load_class_mapping():
    """Load the class mapping generated during training."""

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_indices = json.load(file)

    # Convert:
    # {"fire": 0, "nofire": 1}
    # to:
    # {0: "fire", 1: "nofire"}
    return {
        int(index): class_name
        for class_name, index in class_indices.items()
    }


class_mapping = load_class_mapping()

# Initialize S3 client once per Lambda container
s3 = boto3.client("s3")


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image_bytes):
    """Convert an S3 image into model-ready input."""

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

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

    return image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_wildfire(image_array):
    """Run wildfire classification."""

    score = float(
        model.predict(
            image_array,
            verbose=0,
        )[0][0]
    )

    predicted_index = (
        1 if score >= 0.5 else 0
    )

    predicted_class = class_mapping[
        predicted_index
    ]

    confidence = (
        score
        if predicted_index == 1
        else 1.0 - score
    )

    predicted_class = (
        predicted_class
        .replace("_", " ")
        .title()
    )

    return (
        predicted_class,
        score,
        confidence,
    )


# --------------------------------------------------
# Lambda Handler
# --------------------------------------------------

def lambda_handler(event, context):
    """
    Run wildfire detection on an image stored in S3.

    Expected event:

    {
        "image_bucket": "bucket-name",
        "image_key": "path/to/image.jpg"
    }
    """

    if (
        "image_bucket" not in event
        or "image_key" not in event
    ):
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": (
                        "Please provide "
                        "'image_bucket' and "
                        "'image_key'."
                    )
                }
            ),
            "headers": {
                "Content-Type":
                    "application/json"
            },
        }

    bucket = event["image_bucket"]
    key = event["image_key"]

    try:

        # ------------------------------------------
        # Retrieve Image from Amazon S3
        # ------------------------------------------

        response = s3.get_object(
            Bucket=bucket,
            Key=key,
        )

        image_bytes = (
            response["Body"].read()
        )

        # ------------------------------------------
        # Preprocess Image
        # ------------------------------------------

        image_array = preprocess_image(
            image_bytes
        )

        # ------------------------------------------
        # Run CNN Inference
        # ------------------------------------------

        (
            predicted_class,
            score,
            confidence,
        ) = predict_wildfire(
            image_array
        )

        # ------------------------------------------
        # Return Prediction
        # ------------------------------------------

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "predicted_class":
                        predicted_class,

                    "prediction_score":
                        round(score, 6),

                    "confidence":
                        round(confidence, 6),

                    "image_bucket":
                        bucket,

                    "image_key":
                        key,
                }
            ),
            "headers": {
                "Content-Type":
                    "application/json"
            },
        }

    except Exception as error:

        print(
            f"Wildfire inference failed: "
            f"{error}"
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": (
                        "Unable to process "
                        "image for wildfire "
                        "detection."
                    )
                }
            ),
            "headers": {
                "Content-Type":
                    "application/json"
            },
        }
