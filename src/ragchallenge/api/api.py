from ragchallenge.api.interfaces.database import DocumentStore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import Settings
# from .routers import service
from .routers import qa_service
from .routers import query_service
from .routers import question_service

# =================== Settings ===================

# Load application settings
settings = Settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    contact={
        "name": settings.contact_name,
        "url": settings.contact_url,
        "email": settings.contact_email,
    },
    license_info={
        "name": settings.license_name,
        "url": settings.license_url,
    },
)

# CORS Settings
origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- Include Routers -------------------------- #

# Register sub modules
app.include_router(qa_service.router, tags=["Answer Generation"])
app.include_router(query_service.router, tags=["Query Expansion"])
app.include_router(question_service.router, tags=[
                   "Hypothetical Question Generation"])
