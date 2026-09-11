import json
import sys
from pathlib import Path

import pytest

# tests/ no forma parte del paquete `app`; nos aseguramos de que backend/ (padre de tests/)
# este en sys.path para poder hacer `from app.main import app`.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CLASS_NAMES = ["downdog", "goddess", "plank", "tree", "warrior2"]


@pytest.fixture(scope="session")
def dummy_model_dir(tmp_path_factory):
    """Modelo Keras diminuto con la misma forma de entrada/salida que el real.

    Los tests de contrato de la API no deben depender del modelo de produccion (.h5 de
    decenas de MB, generado en Colab) — solo necesitan que /predict tenga la forma correcta.
    """
    from tensorflow.keras import layers, models

    models_dir = tmp_path_factory.mktemp("models")

    dummy_model = models.Sequential(
        [
            layers.Input(shape=(224, 224, 3)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    dummy_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    dummy_model.save(models_dir / "modelo_yoga_kaggle.h5")

    with open(models_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f)

    return models_dir


@pytest.fixture()
def client(dummy_model_dir, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(dummy_model_dir))

    # Los modulos de la app pueden haber quedado cacheados de una ejecucion anterior con otra
    # MODELS_DIR; se recargan para que app.config lea la variable de entorno de este test.
    for mod_name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod_name]

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
