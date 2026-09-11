import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from . import config


class ModelNotLoadedError(RuntimeError):
    """El modelo aun no fue cargado (falta llamar a PoseClassifier.load())."""


class InvalidImageError(ValueError):
    """El archivo recibido no es una imagen valida."""


class PoseClassifier:
    """Envuelve el modelo Keras entrenado en training/Entrenamiento_Modelo_Yoga.ipynb.

    El preprocesado replica exactamente el del notebook: redimensionar a
    config.IMAGE_SIZE y normalizar dividiendo entre 255.
    """

    def __init__(self):
        self._model = None
        self._class_names: list[str] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            raise ModelNotLoadedError("El modelo aun no ha sido cargado")
        return self._class_names

    def load(self, model_path: Path | None = None, labels_path: Path | None = None) -> None:
        model_path = model_path or config.MODEL_PATH
        labels_path = labels_path or config.LABELS_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {model_path}. Entrena el modelo en Colab "
                "(ver training/README.md) y copia los artefactos a backend/models/."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"No se encontro labels.json en {labels_path}.")

        # Se cargan ambos artefactos en variables locales primero: si labels.json esta corrupto
        # despues de cargar el modelo (o viceversa), el estado de la instancia no debe quedar a
        # medias con is_loaded=True pero class_names en None.
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        model = load_model(model_path)

        self._model = model
        self._class_names = class_names

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        try:
            probe = Image.open(io.BytesIO(image_bytes))
            probe.verify()
        except Exception as exc:
            raise InvalidImageError("El archivo no es una imagen valida") from exc

        # Image.verify() deja el objeto inutilizable para mas operaciones; se reabre.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(config.IMAGE_SIZE)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError("El modelo aun no ha sido cargado")

        batch = self.preprocess(image_bytes)
        probabilities = self._model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        return {
            "pose": self._class_names[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                name: float(prob) for name, prob in zip(self._class_names, probabilities)
            },
        }


# Instancia unica compartida por la app (cargada una vez en el lifespan de FastAPI).
classifier = PoseClassifier()
