from fastapi import FastAPI
from deployment.api.config import APP_NAME, VERSION, DEBUG

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    debug=DEBUG,
    description="API para la detección de noticias falsas.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


@app.get("/", tags=["Root"])
def root():
    return {"message": f"¡Bienvenido a {APP_NAME}!"}
