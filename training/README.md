# Entrenamiento del modelo (Google Colab)

Este notebook entrena el clasificador de posturas de yoga que usa la aplicación
(`backend/` + `frontend/`). Se ejecuta en **Google Colab**, no localmente — usa GPU gratuita y
`kagglehub` para descargar el dataset.

## Pasos

1. Abre [Google Colab](https://colab.research.google.com/) y sube o abre
   `Entrenamiento_Modelo_Yoga.ipynb` (`Archivo → Subir notebook`).
2. Activa GPU: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. Ejecuta todas las celdas en orden (`Entorno de ejecución → Ejecutar todas`).
   - La primera vez te pedirá autenticarte con Kaggle (credenciales de tu cuenta de Kaggle) para
     descargar el dataset `niharika41298/yoga-poses-dataset`.
   - El entrenamiento tiene dos etapas (capas nuevas, luego fine-tuning de MobileNetV2) y se
     detiene automáticamente con `EarlyStopping` cuando deja de mejorar — normalmente toma entre
     10 y 25 minutos en GPU T4.
4. Al final, el notebook genera 3 archivos en `/content/`:
   - `modelo_yoga_kaggle.h5` — modelo entrenado
   - `labels.json` — lista ordenada de clases (`downdog`, `goddess`, `plank`, `tree`, `warrior2`)
   - `model_metrics.json` — accuracy y precisión/recall/F1 por clase, medidos sobre el conjunto de
     **TEST** real (no sobre el split de validación)
5. Descarga los 3 archivos del panel de archivos de Colab (clic derecho → Descargar) y colócalos
   en `backend/models/` de este repositorio, con esos mismos nombres.

## Diferencias con el notebook académico original

Ver la celda "Cambios respecto al notebook original" al inicio del notebook. En resumen: se agrega
data augmentation y callbacks de entrenamiento (EarlyStopping/Checkpoint/ReduceLROnPlateau), una
etapa de fine-tuning, y se corrige la evaluación final para que use el conjunto de TEST real en
vez de reusar el split de validación. También se corrige el *orden* de `.ignore_errors()` (sigue
siendo necesario — el dataset trae al menos un JPEG corrupto — pero se aplica antes de `cache()`
en vez de al final de la cadena).

**Resultado obtenido**: 78.3% de accuracy sobre el conjunto de TEST real (ver
`backend/models/model_metrics.json` una vez entrenado).

## Alternativa: entrenar localmente en vez de Colab

Si prefieres no usar Colab, `training/train_local.py` replica la misma lógica y corre en esta
máquina con CPU (más lento, pero funcional):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r ..\training\requirements.txt
$env:KAGGLE_API_TOKEN = "tu-token-de-kaggle"   # ver https://www.kaggle.com/settings -> API
python ..\training\train_local.py
```

Genera los 3 artefactos directamente en `backend/models/` (no hace falta copiarlos a mano).

## Después de entrenar

Con los 3 archivos en `backend/models/`, sigue las instrucciones de `backend/README.md` para
levantar la API y `frontend/README.md` para la interfaz web.
