# Backend — API de inferencia

API en FastAPI que sirve el modelo entrenado en `training/` y expone un endpoint para clasificar
la postura de yoga presente en una imagen.

## Requisitos

- Python 3.13 (TensorFlow todavía no soporta 3.14). En esta máquina:
  `C:\Users\diana\AppData\Local\Programs\Python\Python313\python.exe`
- El modelo entrenado en `backend/models/` (ver más abajo).

## 1. Colocar el modelo entrenado

Copia estos 3 archivos generados por `training/Entrenamiento_Modelo_Yoga.ipynb` (Colab) a
`backend/models/`:

- `modelo_yoga_kaggle.h5`
- `labels.json`
- `model_metrics.json`

Sin estos archivos, la API arranca igual pero `/predict` y `/classes` responden `503` hasta que
se coloquen.

## 2. Crear entorno virtual e instalar dependencias

```powershell
cd backend
C:\Users\diana\AppData\Local\Programs\Python\Python313\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Ejecutar la API

```powershell
uvicorn app.main:app --reload --port 8000
```

- Documentación interactiva: http://localhost:8000/docs
- Salud del servicio: http://localhost:8000/health

## 4. Ejecutar las pruebas

```powershell
pytest
```

Las pruebas usan un modelo Keras diminuto generado en memoria (ver `tests/conftest.py`), no el
modelo real de producción — así no dependen de tener el `.h5` entrenado para validar el contrato
de la API.

## Endpoints

| Método | Ruta        | Descripción                                              |
|--------|-------------|-----------------------------------------------------------|
| GET    | `/health`   | Estado del servicio y si el modelo está cargado           |
| GET    | `/classes`  | Lista de posturas que el modelo reconoce                  |
| POST   | `/predict`  | Sube una imagen (`multipart/form-data`, campo `file`) y devuelve la postura predicha, confianza y probabilidades por clase |

### Ejemplo de respuesta de `/predict`

```json
{
  "pose": "downdog",
  "confidence": 0.94,
  "probabilities": {
    "downdog": 0.94,
    "goddess": 0.01,
    "plank": 0.02,
    "tree": 0.01,
    "warrior2": 0.02
  }
}
```

## Configuración (variables de entorno opcionales)

- `MODELS_DIR`: carpeta donde buscar `modelo_yoga_kaggle.h5` / `labels.json` (por defecto,
  `backend/models`).
- `CORS_ORIGINS`: orígenes permitidos separados por coma (por defecto incluye
  `http://localhost:5500` y `http://127.0.0.1:5500`, donde suele correr el frontend estático).
