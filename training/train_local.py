"""Entrena el modelo localmente (CPU), como alternativa a Colab.

Replica exactamente la logica de Entrenamiento_Modelo_Yoga.ipynb, para poder
correrla como script en esta maquina cuando no se quiere usar Colab.
Genera modelo_yoga_kaggle.h5, labels.json y model_metrics.json directamente en
backend/models/.
"""
import itertools
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    import kagglehub

    print("Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
    print("Dataset en:", path)

    subfolder = os.path.join(path, os.listdir(path)[0])
    train_path = os.path.join(subfolder, "TRAIN")
    test_path = os.path.join(subfolder, "TEST")
    print("TRAIN:", os.listdir(train_path))
    print("TEST:", os.listdir(test_path))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    print("Clases:", class_names)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        shuffle=False,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    assert test_ds.class_names == class_names

    # El dataset trae al menos un JPEG corrupto (confirmado en la practica: rompe la decodificacion
    # a mitad de entrenamiento). ignore_errors() SI es un metodo valido de tf.data.Dataset -
    # descarta silenciosamente los elementos que fallan al decodificarse, en vez de abortar el fit().
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.ignore_errors().cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.ignore_errors().cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.ignore_errors().cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print("\n--- Etapa 1: entrenando capas nuevas ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

    print("\n--- Etapa 2: fine-tuning de MobileNetV2 ---")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_epochs = 15
    total_epochs = history.epoch[-1] + 1 + fine_tune_epochs
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "modelo_yoga_kaggle.h5")
    with open(MODELS_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print("Modelo y labels guardados en", MODELS_DIR)

    print("\n--- Evaluacion sobre TEST ---")
    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    print("Matriz de confusion:\n", confusion_matrix(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names))
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\nMetricas:", json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
