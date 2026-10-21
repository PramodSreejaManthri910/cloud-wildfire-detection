import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = os.path.join(
    "data",
    "Training and Validation",
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "wildfire_model.keras",
)

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Data Preprocessing and Augmentation
# --------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
)


validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)


# --------------------------------------------------
# CNN Model
# --------------------------------------------------

def build_model():
    """Build a CNN for binary wildfire image classification."""

    model = Sequential(
        [
            tf.keras.Input(shape=(150, 150, 3)),

            Conv2D(
                32,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(2, 2),

            Conv2D(
                64,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(2, 2),

            Conv2D(
                128,
                (3, 3),
                activation="relu",
            ),
            MaxPooling2D(2, 2),

            Flatten(),

            Dense(
                128,
                activation="relu",
            ),
            Dropout(0.5),

            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def main():

    model = build_model()

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
    )

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            checkpoint,
        ],
    )

    print(
        f"\nTraining complete. "
        f"Best model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
