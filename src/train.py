# train.py
# Train a RetinaNet detector using KerasCV on fire/smoke dataset.

import os
import tensorflow as tf
from tensorflow import keras
import keras_cv

from src.config import MODEL_DIR, MODEL_PATH, NUM_CLASSES, BBOX_FORMAT
from src.dataset_prep import get_train_val_datasets


AUTOTUNE = tf.data.AUTOTUNE


def create_model():
    """Create RetinaNet with ResNet50 backbone."""
    model = keras_cv.models.RetinaNet.from_preset(
        "resnet50_imagenet",
        num_classes=NUM_CLASSES,
        bounding_box_format=BBOX_FORMAT,
    )
    return model


def train(epochs=1, batch_size=4, lr=1e-4):
    # Datasets return dict: {"images": ..., "bounding_boxes": {...}}
    train_ds, val_ds = get_train_val_datasets(batch_size=batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    model = create_model()

    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        classification_loss="focal",
        box_loss="smoothl1",
        jit_compile=False,
    )

    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "checkpoint.keras"),
            save_weights_only=False,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ❌ DO NOT DO THIS: model.save(MODEL_DIR)
    # ✅ Save to a file with .keras extension
    model.save(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    train(epochs=10, batch_size=4, lr=1e-4)
