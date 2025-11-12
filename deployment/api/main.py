from fastapi import FastAPI
from deployment.api.routes import router
from fastapi.middleware.cors import CORSMiddleware
from deployment.api.config import APP_NAME, VERSION, DEBUG

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    debug=DEBUG,
    description="""
    ## 🔍 API para Detección de Noticias Falsas
    
    Esta API utiliza modelos de Machine Learning y Deep Learning para clasificar 
    noticias como reales o falsas.
    
    ### Modelos Disponibles:
    * **Naive Bayes**: Clasificador probabilístico con TF-IDF
    * **Random Forest**: Ensamble de árboles de decisión
    * **CNN**: Red neuronal convolucional
    * **Híbrido**: CNN + BiLSTM + GRU con atención
    
    ### Características:
    * Predicción en tiempo real
    * Múltiples modelos de ML/DL
    * Métricas de rendimiento
    * Sistema de logs
    * Monitoreo de salud
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
        "syntaxHighlight.theme": "monokai",
    },
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    """
    Endpoint raíz de bienvenida.
    """
    return {
        "message": f"¡Bienvenido a {APP_NAME}!",
        "version": VERSION,
        "docs": "/docs",
        "api": "/api/v1",
    }


@app.get("/ping", tags=["Root"])
def ping():
    """
    Endpoint simple para verificar conectividad.
    """
    return {"status": "pong"}
