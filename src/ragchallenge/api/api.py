from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import Settings
from .routers import service

# =================== Settings ===================

# Load application settings
settings = Settings()

tags_metadata = [
    {
        "name": "service",
        "description": "Entity Linking",
    },
]
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
    openapi_tags=tags_metadata,
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

# Register sub modules
app.include_router(service.router, tags=["service"])
