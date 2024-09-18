from ragchallenge.api.utils.documentstore import DocumentStore
from ragchallenge.api.config import settings

# ---------------------------- Load Database --------------------------- #

DATABASE = DocumentStore(model_name=settings.embedding_model,
                         persist_directory=settings.data_dir,
                         device=settings.embedding_model_device)
