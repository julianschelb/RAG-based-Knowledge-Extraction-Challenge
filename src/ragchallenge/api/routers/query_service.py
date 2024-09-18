from fastapi import APIRouter, HTTPException
from ragchallenge.api.paraphraser import PARAPHRASER
from ragchallenge.api.schemas.messages import ChatRequest, QueryResponse

router = APIRouter(responses={404: {"description": "Not Found"}})


# ---------------------------- Endpoints --------------------------- #

@router.post("/expand-query", response_model=QueryResponse)
async def generate_queries(request: ChatRequest):
    """Generate expanded and hypothetical queries for a given question. Is used to expand the user's input."""
    try:
        # Get the user's question from the last message in the list and expand it
        user_message = request.messages[-1].content
        expanded_queries = PARAPHRASER.rephrase(user_message)
        return QueryResponse(original_query=user_message, expanded_queries=expanded_queries)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
