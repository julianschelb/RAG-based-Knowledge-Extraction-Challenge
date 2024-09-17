from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "API Name"
    app_description: str = "API Description"
    app_version: str = "1.0"
    contact_name: str = "Contact Name"
    contact_email: str = "test@contact.email"
    contact_url: str = "https://www.test.contact.url/"
    license_name: str = ""
    license_url: str = ""
    data_dir: str = "data"

    class Config:
        env_file = ".env"


settings = Settings()
