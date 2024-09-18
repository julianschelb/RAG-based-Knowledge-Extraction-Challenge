from ragchallenge.api.interfaces.database import DocumentStore
from ragchallenge.api.config import settings

# ---------------------------- Load Database --------------------------- #

DATABASE = DocumentStore(model_name=settings.embedding_model,
                         persist_directory="data/vectorstore_augmented",  # settings.data_dir,
                         device=settings.embedding_model_device)

# print(settings.data_dir)
# print(settings.embedding_model)
print(len(DATABASE.vector_store.get()["ids"]))
