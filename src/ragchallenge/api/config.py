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
    embedding_model: str = ""
    embedding_model_device: str = "cpu"
    chat_model: str = ""
    chat_model_task: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
