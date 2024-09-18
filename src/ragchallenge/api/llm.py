from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from ragchallenge.api.config import settings

# ---------------------------- Load Model ---------------------------

# Parameters for generation (you can adjust these as needed)
generation_params = {
    "temperature": 0.7,
    "max_new_tokens": 230,
    "max_new_tokens": 2048,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
}

# Create the Hugging Face Endpoint using the specified parameters
endpoint = HuggingFaceEndpoint(
    repo_id=settings.chat_model,
    task=settings.chat_model_task,
    **generation_params,
)

# Return the LangChain HuggingFacePipeline object with the endpoint
LLM = ChatHuggingFace(llm=endpoint)
