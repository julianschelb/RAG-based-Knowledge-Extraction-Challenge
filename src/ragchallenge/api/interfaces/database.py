import os
import re
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from transformers import BertTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class DocumentStore:
    """Class to load, process, split documents, and create a vector store."""

    def __init__(self, model_name: str = "thenlper/gte-small", device="mps", persist_directory: str = "../data/vectorstore"):
        """Initialize the DocumentStore class with an embedding model and Chroma vector store."""

        # Initialize the HuggingFaceEmbeddings model
        self.embedding_model = HuggingFaceEmbeddings(
            # multi_process=True,
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
        )

        # Initialize the Chroma vector store
        self.vector_store = Chroma(
            collection_name="documentation",
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", clean_up_tokenization_spaces=True)

    def validate_directory(self, directory_path: str) -> None:
        """Validate the directory path."""
        if not isinstance(directory_path, str):
            raise ValueError("The directory path must be a string.")
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(
                f"The specified directory '{directory_path}' does not exist.")
        if not os.access(directory_path, os.R_OK):
            raise PermissionError(
                f"Cannot read from the directory: {directory_path}")

    def load_markdown_documents(self, directory_path: str) -> List[Document]:
        """Load all markdown documents from a directory."""
        self.validate_directory(directory_path)
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.md",
                loader_cls=TextLoader
            )
            documents = loader.load()
            if not documents:
                raise ValueError(
                    "No documents were loaded from the specified directory.")
            return documents
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while loading documents: {str(e)}")

    def clean_string(self, text: str) -> str:
        """Remove special characters and return the cleaned string."""
        text = re.sub(r'\s+', ' ', text).strip()
        return re.sub(r'[^a-zA-Z ]', '', text)

    def clean_source(self, path: str) -> str:
        """Extract the filename from the path, remove the extension, replace non-alpha with spaces, and clean it."""
        filename = os.path.splitext(os.path.basename(path))[0]
        cleaned_filename = re.sub(r'[^a-zA-Z]', ' ', filename)
        return re.sub(r'\s+', ' ', cleaned_filename).strip()

    def split_single_document(self, document: Document, header: str = "##") -> List[Document]:
        """Split a single document by a given header, treating everything before the first header as its own document."""
        split_docs = []
        content = document.page_content

        # Split the content by the header
        sections = content.split(header)

        # Treat everything before the first header as its own document
        if sections[0].strip():
            pre_header_content = sections[0].strip()
            pre_header_metadata = {
                **document.metadata,
                "title": "" if "title" not in document.metadata else document.metadata["title"],
                "cleaned_title": self.clean_string("") if "title" not in document.metadata else self.clean_string(document.metadata["title"]),
                "cleaned_source": self.clean_source(document.metadata.get("source", "")),
            }
            pre_header_doc = document.model_copy(
                update={"page_content": pre_header_content, "metadata": pre_header_metadata})
            split_docs.append(pre_header_doc)

        # Process each section after the first header
        for section in sections[1:]:
            try:
                title, *body = section.split('\n', 1)
                body_content = body[0].lstrip() if body else ''
                new_metadata = {
                    **document.metadata,
                    "title": title.strip(),
                    "cleaned_title": self.clean_string(title.strip()),
                    "cleaned_source": self.clean_source(document.metadata.get("source", "")),
                }
                new_doc = document.model_copy(
                    update={"page_content": body_content,
                            "metadata": new_metadata}
                )
                split_docs.append(new_doc)
            except Exception as e:
                print(f"Error processing section: {e}")

        return split_docs

    def split_documents_by_header(self, documents: List[Document], header: str = "##") -> List[Document]:
        """Split a list of documents by a given header."""
        return [doc for document in documents for doc in self.split_single_document(document, header)]

    def get_length(self, text: str) -> int:
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def filter_documents_by_token_length(self, documents: List[Document], min_token_length: int = 25) -> List[Document]:
        """
        Filter out documents that have fewer than the specified minimum number of tokens.

        :param documents: List of Document objects.
        :param tokenizer: A tokenizer to tokenize the text.
        :param min_token_length: Minimum number of tokens required to include the document.
        :return: List of Document objects that meet the token length requirement.
        """

        # Use list comprehension to filter documents based on token length
        return [doc for doc in documents if self.get_length(doc.page_content) >= min_token_length]

    def split_documents_by_token_count(self, documents: List[Document], chunk_size: int = 256, chunk_overlap: int = 192) -> List[Document]:
        """Splits documents using BERT token count."""

        custom_separators = ["\n\n", "\n", ".", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.get_length,
            is_separator_regex=False,
            separators=custom_separators
        )
        return text_splitter.split_documents(documents)

    def add_documents_to_vector_store(self, documents: List[Document]) -> None:
        """Add documents to the Chroma vector store."""
        self.vector_store.add_documents(documents)

    def process_and_add_documents(database, documents: List[Document], header: str = "##", chunk_size: int = 192, chunk_overlap: int = 64) -> None:
        """
        Split documents by header, chunk them by token count, and add the resulting documents to the vector store.

        :param database: The database object that has methods to split and add documents to the vector store.
        :param documents: List of Document objects to be processed.
        :param header: The header by which to split the documents.
        :param chunk_size: The maximum number of tokens in each chunk.
        :param chunk_overlap: The number of tokens to overlap between chunks.
        """
        # Split the documents by header
        split_documents = database.split_documents_by_header(
            documents, header=header)
        print("Number of documents after splitting by header: ",
              len(split_documents))

        # Chunk the split documents by token count
        documents_chunked = database.split_documents_by_token_count(
            split_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print("Number of documents after chunking: ", len(documents_chunked))

        # Add the chunked documents to the vector store
        database.add_documents_to_vector_store(documents_chunked)

    def query_vector_store(self, query: str, k: int = 5) -> List[Document]:
        """Query the vector store using a user query and return the top-k results."""
        # query_vector = self.embedding_model.embed_query(query)
        return self.vector_store.similarity_search(query=query, k=k)


# Example usage of the class
if __name__ == "__main__":
    processor = DocumentStore()  # Assuming your class is named DocumentProcessor

    # Manually create two Document objects
    document1 = Document(
        page_content="Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.",
        metadata={"title": "Introduction to Conda", "source": "manual"}
    )

    document2 = Document(
        page_content="To start Conda, first install it by downloading the Anaconda or Miniconda distribution. After installation, open a terminal and run 'conda'.",
        metadata={"title": "How to Start Conda", "source": "manual"}
    )

    document3 = Document(
        page_content="""
        # Installation

        To install Conda, you need to download either the Anaconda or Miniconda distribution. Choose the one that suits your needs.

        ## Anaconda

        Anaconda is a full-featured distribution, including many pre-installed packages like NumPy, pandas, and others.

        ## Miniconda

        Miniconda is a minimal version of Anaconda, allowing you to install only the packages you need.

        # Usage

        Once installed, you can use Conda to create new environments by running 'conda create --name myenv'.
        """,
        metadata={"title": "Conda Installation and Usage", "source": "manual"}
    )

    # List of manually created documents
    documents = [document1, document2, document3]

    # Process and split documents by header (if necessary) and by token count
    split_documents = processor.split_documents_by_header(documents)
    documents_chunked = processor.split_documents_by_token_count(
        split_documents)

    print(
        f"Loaded {len(documents)} documents, split into {len(documents_chunked)} smaller documents")

    # Add documents to vector store
    processor.add_documents_to_vector_store(documents_chunked)

    # Query the vector store
    user_query = "How to start conda?"
    results = processor.query_vector_store(user_query)

    # Print results
    for result_id, result in enumerate(results):
        print(
            f"\n============================== Document {result_id + 1} ==============================")
        print(result.metadata)
        print(result.page_content)
