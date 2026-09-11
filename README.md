# Sistema de Reconocimiento de Posturas de Yoga

Aplicación que identifica la postura de yoga presente en una foto, entre 5 posturas: **Perro boca
abajo** (downdog), **Diosa** (goddess), **Plancha** (plank), **Árbol** (tree) y **Guerrero II**
(warrior2).

Nace de la actividad académica "Evaluemos un modelo de clasificación" (Maestría en IA — Visión
por Computador): el notebook original entrena un clasificador de imágenes con transfer learning
(MobileNetV2) sobre el dataset
[`niharika41298/yoga-poses-dataset`](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset)
de Kaggle. Este repositorio convierte ese notebook en una aplicación completa: API de inferencia +
interfaz web para subir una foto y ver la predicción.

## Arquitectura

```
Usuario → frontend/ (HTML/CSS/JS) → backend/ (FastAPI + modelo Keras) → predicción
                                              ↑
                              modelo entrenado en training/ (Google Colab)
```

- **`training/`** — notebook de Colab que descarga el dataset, entrena el modelo (MobileNetV2 +
  fine-tuning) y exporta `modelo_yoga_kaggle.h5`, `labels.json` y `model_metrics.json`.
- **`backend/`** — API en FastAPI que carga el modelo entrenado y expone `POST /predict` para
  clasificar una imagen subida.
- **`frontend/`** — página web estática (sin frameworks) que sube la imagen, llama a la API y
  muestra el resultado con la confianza y probabilidades por clase.
- **`samples/`** — imágenes de ejemplo para probar la app manualmente.

## Puesta en marcha (de cero a la app funcionando)

1. **Entrenar el modelo** — en Colab (GPU, recomendado) o localmente con `training/train_local.py`
   (CPU, más lento). Sigue `training/README.md`. Al final tendrás 3 archivos:
   `modelo_yoga_kaggle.h5`, `labels.json`, `model_metrics.json`.
2. **Colocar el modelo** — copia esos 3 archivos a `backend/models/`.
3. **Levantar la API** — sigue `backend/README.md` (crear venv con Python 3.13, instalar
   dependencias, `uvicorn app.main:app --reload --port 8000`).
4. **Levantar el frontend** — sigue `frontend/README.md` (`python -m http.server 5500` dentro de
   `frontend/`), abre http://localhost:5500.
5. Sube una imagen de `samples/` (o cualquier foto de una de las 5 posturas) y presiona "Analizar
   postura".

El único paso manual que no se puede automatizar desde aquí es el paso 1-2: entrenar en Colab
requiere GPU y credenciales de Kaggle, así que se hace fuera de este entorno y luego se copian los
artefactos.

## Diferencias con el notebook académico original

El notebook original (conservado en
`training/ACTIVIDAD_5__Evaluemos_un_modelo_de_clasificación_original.ipynb` para referencia) tenía
varios problemas de rigor de ML que se corrigieron en `training/Entrenamiento_Modelo_Yoga.ipynb`:
entrenamiento sin control (100 épocas fijas, sin early stopping) y evaluación final reusando el
split de validación en vez de un conjunto de TEST real. El detalle completo está en la primera
celda de ese notebook y en `training/README.md`.

El modelo actual de este repositorio se entrenó localmente (`training/train_local.py`, alternativa
a Colab) y alcanzó **78.3% de accuracy** sobre el conjunto de TEST real — el detalle por clase está
en `backend/models/model_metrics.json`. La postura más difícil de reconocer es "Diosa" (goddess),
igual que en el notebook académico original.

## API — referencia rápida

| Método | Ruta       | Descripción                                    |
|--------|------------|-------------------------------------------------|
| GET    | `/health`  | Estado del servicio y si el modelo está cargado |
| GET    | `/classes` | Lista de posturas reconocidas                   |
| POST   | `/predict` | Sube una imagen y devuelve la predicción         |

Documentación interactiva completa en `http://localhost:8000/docs` una vez que la API está
corriendo. Detalles en `backend/README.md`.

## Pruebas

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest
```

Las pruebas de la API usan un modelo Keras diminuto generado en memoria (no el modelo real de
producción), por lo que no requieren haber entrenado en Colab para poder correrlas.

## Limitaciones conocidas

- Solo reconoce las 5 posturas del dataset de entrenamiento — cualquier otra postura se clasifica
  igual como la más parecida de esas 5 (el modelo no tiene una clase "ninguna" ni umbral de
  rechazo).
- Clasifica **imágenes estáticas**, no video en tiempo real ni landmarks corporales (no usa
  MediaPipe ni similares).
- El dataset de Kaggle tiene su propia licencia de uso — revisar antes de cualquier uso comercial.

## Estructura del repositorio

```
├── README.md
├── samples/            # imágenes de ejemplo
├── training/            # notebook de entrenamiento (Colab) + su README
├── backend/              # API FastAPI (código, tests, README)
└── frontend/             # interfaz web estática (HTML/CSS/JS, README)
```
<img width="1066" height="591" alt="image" src="https://github.com/user-attachments/assets/49559b15-d53e-49f1-9464-e57ecb907dd9" />
