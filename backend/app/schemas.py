from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    pose: str = Field(..., description="Clase de postura predicha")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la clase predicha (0-1)")
    probabilities: dict[str, float] = Field(
        ..., description="Probabilidad asignada a cada una de las clases"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str] = []
