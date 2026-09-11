from pathlib import Path

from tests.conftest import CLASS_NAMES

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


def _sample_image_bytes() -> bytes:
    sample = next(SAMPLES_DIR.glob("*.jpg"))
    return sample.read_bytes()


def test_predict_returns_valid_prediction(client):
    files = {"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["pose"] in CLASS_NAMES
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"].keys()) == set(CLASS_NAMES)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3


def test_predict_rejects_unsupported_content_type(client):
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_rejects_corrupted_image_bytes(client):
    files = {"file": ("fake.jpg", b"not-a-real-jpeg-just-bytes", "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_rejects_oversized_file(client):
    huge = b"0" * (10 * 1024 * 1024 + 1)
    files = {"file": ("big.jpg", huge, "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_rejects_empty_file(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400
