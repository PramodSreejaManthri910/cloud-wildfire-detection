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

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "wildfire_model.keras",
)


# --------------------------------------------------
# Load Resources
# --------------------------------------------------

# Load once during Lambda container initialization
# instead of reloading the model for every request.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

s3 = boto3.client("s3")


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image_bytes):
    """Convert S3 image bytes into model-ready input."""

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

    # Preserve the class convention used by
    # the original trained wildfire model.
    if score < 0.5:
        predicted_class = "Fire"
        confidence = 1.0 - score
    else:
        predicted_class = "No Fire"
        confidence = score

    return predicted_class, score, confidence


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
                        "Please provide 'image_bucket' "
                        "and 'image_key'."
                    )
                }
            ),
            "headers": {
                "Content-Type": "application/json"
            },
        }

    bucket = event["image_bucket"]
    key = event["image_key"]

    try:
        # Retrieve satellite/image data from Amazon S3
        response = s3.get_object(
            Bucket=bucket,
            Key=key,
        )

        image_bytes = response["Body"].read()

        # Preprocess image
        image_array = preprocess_image(
            image_bytes
        )

        # Run inference
        predicted_class, score, confidence = (
            predict_wildfire(image_array)
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "predicted_class": predicted_class,
                    "prediction_score": round(
                        score,
                        6,
                    ),
                    "confidence": round(
                        confidence,
                        6,
                    ),
                    "image_bucket": bucket,
                    "image_key": key,
                }
            ),
            "headers": {
                "Content-Type": "application/json"
            },
        }

    except Exception as error:
        print(
            f"Wildfire inference failed: {error}"
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": (
                        "Unable to process image "
                        "for wildfire detection."
                    )
                }
            ),
            "headers": {
                "Content-Type": "application/json"
            },
        }
