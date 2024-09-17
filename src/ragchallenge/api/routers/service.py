# ---------------------------------------------------------------------------- #
#                      Language Model Service                 #
# ---------------------------------------------------------------------------- #
# This section contains API entpoints for generating responses from the model.

from fastapi import APIRouter, HTTPException, status  # , Body, status, Depends
from typing import List
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse, Response
# from fastapi import HTTPException, status
# from app.schemas.userRequest import UserRequest, SystemResponse

router = APIRouter(responses={404: {"description": "Not Found"}})

# ---------------------------- Helpers --------------------------- #


@router.get("/hello")
async def hello_world():
    return {"message": "Hello World"}

# ---------------------------- Endpoints --------------------------- #


@router.post("/documents/", status_code=status.HTTP_201_CREATED)
# You can define your document schema later
async def add_new_document(document: str):
    """
    Endpoint to add a new document to the system.

    Args:
        document (str): The document text to be added.

    Returns:
        The status of the operation.
    """
    pass


@router.get("/documents/", status_code=status.HTTP_200_OK)
# Adjust return type based on your data structure
async def retrieve_relevant_documents(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve relevant documents based on the query.

    Args:
        query (str): The search query to retrieve relevant documents.
        top_k (int): Number of top documents to retrieve, default is 5.

    Returns:
        List of relevant documents.
    """
    pass


@router.post("/generate-answer/", status_code=status.HTTP_200_OK)
# Adjust return type based on your data structure
async def generate_answer(question: str) -> str:
    """
    Generate an answer to a given question using the system's model.

    Args:
        question (str): The question for which an answer will be generated.

    Returns:
        The generated answer.
    """
    pass


@router.post("/expand-query/", status_code=status.HTTP_200_OK)
# Adjust return type based on your requirements
async def expand_query(query: str) -> str:
    """
    Expand the given query for better retrieval.

    Args:
        query (str): The query that needs to be expanded.

    Returns:
        The expanded query.
    """
    pass


@router.post("/generate-hypothetical-question/", status_code=status.HTTP_200_OK)
# Adjust return type based on your data structure
async def generate_hypothetical_question(context: str) -> str:
    """
    Generate a hypothetical question based on the given context.

    Args:
        context (str): The context for which a hypothetical question should be generated.

    Returns:
        The generated hypothetical question.
    """
    pass
