from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .inference import InvalidImageError, ModelNotLoadedError, classifier
from .schemas import HealthResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load()
    except FileNotFoundError as exc:
        # La API arranca igual; /predict y /classes devuelven 503 hasta que se coloque el modelo
        # entrenado en backend/models/ (ver training/README.md).
        print(f"[startup] Modelo no disponible todavia: {exc}")
    yield


app = FastAPI(
    title="Sistema de Reconocimiento de Posturas de Yoga - API",
    description="API de inferencia para clasificar posturas de yoga a partir de una imagen.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
        classes=classifier.class_names if classifier.is_loaded else [],
    )


@app.get("/classes", response_model=list[str])
async def classes() -> list[str]:
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado todavia.")
    return classifier.class_names


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado todavia.")

    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Tipo de archivo no soportado.")

    contents = await _read_within_limit(file, config.MAX_UPLOAD_BYTES)
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo esta vacio.")

    try:
        result = classifier.predict(contents)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(**result)


async def _read_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Lee el archivo en bloques, abortando en cuanto se supera el limite en vez de bufferizar
    todo el contenido primero y recien ahi comprobar el tamano."""
    chunk_size = 1024 * 1024
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=400, detail="El archivo supera el tamano maximo permitido (10MB)."
            )
        chunks.append(chunk)

    return b"".join(chunks)
