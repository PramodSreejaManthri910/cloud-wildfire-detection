#  Cloud-Based Wildfire Detection

A deep learning and cloud-based wildfire detection system designed to classify images as **Fire** or **No Fire** using a Convolutional Neural Network (CNN). The project combines image preprocessing, deep learning, and AWS cloud services to demonstrate an end-to-end workflow from model training to cloud-based inference.

## Project Overview

Early wildfire detection can help support faster monitoring and response. This project applies deep learning-based image classification to identify potential wildfire presence from imagery.

The system includes:

- Image preprocessing and data augmentation
- CNN-based binary image classification
- Local wildfire prediction
- Amazon S3 integration for cloud image storage
- AWS Lambda for serverless model inference
- Docker-based Lambda deployment
- JSON-based prediction responses

## Architecture

```text
                    MODEL TRAINING
                         │
                         ▼
                Wildfire Image Dataset
                         │
                         ▼
               Image Preprocessing
                & Data Augmentation
                         │
                         ▼
                  CNN Training
                         │
                         ▼
              Trained Wildfire Model
                         │
                         ▼
                Fire / No Fire


                   CLOUD INFERENCE
                         │
                         ▼
                 Image in Amazon S3
                         │
                         ▼
                    AWS Lambda
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
              TensorFlow CNN Model
                         │
                         ▼
                 Model Inference
                         │
                         ▼
          Fire / No Fire + Confidence
                         │
                         ▼
                  JSON Response
```

## Deep Learning Model

The wildfire detector uses a **Convolutional Neural Network (CNN)** for binary image classification.

The model contains:

- 3 convolutional layers
- Max pooling layers
- ReLU activation
- Fully connected dense layer
- Dropout for regularization
- Sigmoid output for binary classification

Input images are resized to:

```text
150 × 150 × 3
```

The final model predicts one of two classes:

```text
Fire
No Fire
```

## Data Preprocessing

Training images are normalized and augmented to improve model generalization.

The augmentation pipeline includes:

- Rescaling
- Random rotation
- Width shifting
- Height shifting
- Zooming
- Horizontal flipping

The training data is automatically divided into:

```text
80% Training
20% Validation
```

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pillow**
- **Matplotlib**
- **Amazon S3**
- **AWS Lambda**
- **Boto3**
- **Docker**

## Project Structure

```text
cloud-wildfire-detection/
│
├── train_model.py
├── predict.py
├── lambda_function.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── README.md
│
└── models/
    └── wildfire_model.keras
        (generated after training)
```

The trained model and image dataset are not stored in GitHub because they are generated or potentially large artifacts.

## Dataset Setup

Place the wildfire image dataset inside:

```text
data/Training and Validation/
```

Expected structure:

```text
data/
└── Training and Validation/
    ├── fire/
    │   ├── image1.jpg
    │   └── ...
    │
    └── nofire/
        ├── image1.jpg
        └── ...
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PramodSreejaManthri910/cloud-wildfire-detection.git
cd cloud-wildfire-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

After placing the dataset in the expected directory, run:

```bash
python train_model.py
```

The script performs image augmentation, creates the training and validation datasets, trains the CNN, applies early stopping, and saves the best-performing model.

The generated model is stored at:

```text
models/wildfire_model.keras
```

## Local Wildfire Prediction

To classify an image locally:

```bash
python predict.py path/to/image.jpg
```

The script loads the trained model, preprocesses the image, predicts the class, and displays the result with a confidence score.

Example output:

```text
Wildfire Detection Result
------------------------------
Predicted Class : Fire
Model Score     : 0.1421
Confidence      : 85.79%
```

## AWS Cloud Inference

The project also supports serverless inference using **Amazon S3 and AWS Lambda**.

The Lambda function expects an event containing the S3 location of an image:

```json
{
    "image_bucket": "your-bucket-name",
    "image_key": "images/wildfire-image.jpg"
}
```

The inference workflow is:

```text
S3 Image
   ↓
Lambda retrieves image with Boto3
   ↓
Resize and normalize image
   ↓
TensorFlow model inference
   ↓
Fire / No Fire prediction
   ↓
JSON response
```

An example response is:

```json
{
    "predicted_class": "Fire",
    "prediction_score": 0.1421,
    "confidence": 0.8579,
    "image_bucket": "your-bucket-name",
    "image_key": "images/wildfire-image.jpg"
}
```

## Docker Deployment

The AWS Lambda inference environment can be packaged as a Docker container.

Before building the image, train the model so that the following file exists:

```text
models/wildfire_model.keras
```

Build the Docker image:

```bash
docker build -t wildfire-detection .
```

The container uses the AWS Lambda Python runtime and includes the trained TensorFlow model and Lambda inference handler.

The resulting image can be pushed to **Amazon Elastic Container Registry (ECR)** and used to create an AWS Lambda container-based function.

## AWS Services

### Amazon S3

Stores imagery used for cloud-based wildfire inference.

### AWS Lambda

Runs the TensorFlow inference pipeline without requiring a continuously running server.

### Amazon ECR

Can store the Docker image used to deploy the Lambda function.

## Future Improvements

Potential extensions include:

- Integrating satellite imagery pipelines such as MODIS, Sentinel-2, or Landsat
- Automated ingestion of newly available imagery
- Geospatial wildfire visualization
- Adding location metadata to predictions
- Wildfire risk forecasting using weather and environmental data
- API Gateway integration for external applications
- CloudWatch monitoring and logging
- Automated alert generation for high-confidence wildfire detections

## Author

**Pramod Sreeja Manthri**

GitHub: [PramodSreejaManthri910](https://github.com/PramodSreejaManthri910)
