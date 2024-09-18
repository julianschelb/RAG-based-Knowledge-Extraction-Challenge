
from ragchallenge.api.rag import RAG_MODEL
from fastapi import APIRouter, HTTPException
from ragchallenge.api.schemas.messages import ChatResponse, ChatRequest, ChatMessage

router = APIRouter(responses={404: {"description": "Not Found"}})


# ---------------------------- Endpoints --------------------------- #


@router.post("/generate-answer", response_model=ChatResponse)
async def generate_answer(request: ChatRequest):
    """Generate an answer to a question and append it as the last message. Uses RAG, query expansion, and hypothetical question generation."""
    try:
        # Get the user's question from the last message in the list
        user_message = request.messages[-1].content
        response = RAG_MODEL.answer_question(user_message)
        request.messages.append(ChatMessage(role="system", content=response.get("answer")))

        # Return the updated messages list with the generated answer appended
        return ChatResponse(messages=request.messages, questions=response.get("question"), documents=response.get("documents"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
