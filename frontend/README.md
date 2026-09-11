# Frontend — interfaz web

Página estática (HTML/CSS/JS, sin build tooling) que consume la API de `backend/`.

## Ejecutar

Con el backend corriendo en `http://localhost:8000` (ver `backend/README.md`), sirve esta carpeta
con cualquier servidor estático. Por ejemplo, con Python:

```powershell
cd frontend
python -m http.server 5500
```

Abre http://localhost:5500 en el navegador.

> No abras `index.html` con doble clic (`file://`) — el navegador bloquea la petición `fetch` al
> backend por CORS. Debe servirse por HTTP, aunque sea con un servidor simple como el de arriba.

## Configuración

La URL de la API está definida en `js/app.js` (constante `API_BASE_URL`, por defecto
`http://localhost:8000`). Si despliegas el backend en otra URL, actualiza esa constante.

Si sirves el frontend en un puerto distinto de `5500`, agrégalo a `CORS_ORIGINS` en el backend
(ver `backend/README.md`).
