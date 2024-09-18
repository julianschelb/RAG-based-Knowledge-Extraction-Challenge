# ---------------------------------------------------------------------------- #
#                      Language Model Service                 #
# ---------------------------------------------------------------------------- #
# This section contains API entpoints for generating responses from the model.

from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status  # , Body, status, Depends
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from typing import List
import os
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse, Response
# from fastapi import HTTPException, status
# from app.schemas.userRequest import UserRequest, SystemResponse

from ragchallenge.api.interfaces.ragmodel import QuestionAnsweringWithRAG
from ragchallenge.api.interfaces.ragmodelexpanded import QuestionAnsweringWithQueryExpansion
from ragchallenge.api.interfaces.database import DocumentStore
from ragchallenge.api.interfaces.generator import HypotheticalQuestionGenerator

router = APIRouter(responses={404: {"description": "Not Found"}})

# ---------------------------- Helpers --------------------------- #


@router.get("/hello")
async def hello_world():
    return {"message": "Hello World"}

# ---------------------------- Endpoints --------------------------- #


# ---------------------------- Load Database --------------------------- #

database = DocumentStore(model_name="thenlper/gte-small",
                         persist_directory="./data/vectorstore",
                         device="mps")

# ---------------------------- Load Paraphraser --------------------------- #

# Define the prompt template to generate hypothetical questions
messages_hypothetical = [
    SystemMessage(
        role="system",
        content="Generate 3 hypothetical questions based on the following text. "
        "The results should be formatted as a list, with each question separated by a newline."
    ),
    HumanMessage(
        role="user",
        content="Here is the text: {text}\n"
        "Generate 3 hypothetical questions about the above text."
    ),
]

# Create the ChatPromptTemplate from the messages
prompt_template_hypothetical = ChatPromptTemplate.from_messages(
    [(msg.role, msg.content) for msg in messages_hypothetical]
)

# Define the Hugging Face model to use for generating hypothetical questions
repo_id = "HuggingFaceH4/zephyr-7b-beta"  # Model ID from Hugging Face
task = "text-generation"  # Task type

# Parameters for generation (you can adjust these as needed)
generation_params = {
    "temperature": 0.7,
    "max_length": 512,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

# Create the Hugging Face Endpoint using the specified parameters
endpoint = HuggingFaceEndpoint(
    repo_id=repo_id,
    task=task,
    **generation_params,  # Pass the generation parameters
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Return the LangChain HuggingFacePipeline object with the endpoint
llm = ChatHuggingFace(llm=endpoint)

generator = HypotheticalQuestionGenerator(
    model=llm, prompt_template=prompt_template_hypothetical)

# ---------------------------- Load Model --------------------------- #

messages = [
    SystemMessage(
        role="system",
        content="""You are an expert technical assistant specializing in tools like Git, Conda, and regular expressions (Regex).
            Your task is to provide accurate, concise, and context-specific answers to technical questions.
            Format your responses using markdown when necessary, including code snippets or examples where appropriate.

            - Respond only to the question asked, and ensure your answer is precise and relevant to the user's query.
            - If the context is insufficient to answer the question, reply with: 'I don't have enough information to answer that question based on the context provided.'
            - Use bullet points, headings, and code blocks to improve readability where applicable.
            - Keep the explanations clear and straight to the point."""
    ),
    HumanMessage(
        role="user",
        content="""Context:
            {context}
            ---
            Now, here is the question you need to answer:

            Question: {question}"""
    ),
]

# Create the ChatPromptTemplate object
prompt_template = ChatPromptTemplate.from_messages(
    [(msg.role, msg.content) for msg in messages]
)

# Load your language model (replace with actual Hugging Face model)
repo_id = "HuggingFaceH4/zephyr-7b-beta"
task = "text-generation"

generation_params = {
    # Increase temperature slightly to allow some diversity with sampling enabled.
    "temperature": 1.0,
    # Keep max_length for overall response length.
    "max_new_tokens": 2048,
    # Lower top_p to prioritize high-probability tokens.
    "top_p": 0.85,
    # Mild repetition penalty to avoid repetitive phrases while maintaining clarity.
    "repetition_penalty": 1.1,
    # Enable sampling to allow for more flexible and creative responses.
    "do_sample": True,
}

# Create an instance of the Hugging Face model
model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id=repo_id,
    task=task,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    **generation_params
))

# Assume knowledge_vector_database is already defined, if not use None
knowledge_vector_database = database.vector_store
question_generator = generator

# Instantiate the QuestionAnsweringWithRAG class once
rag_chain = QuestionAnsweringWithQueryExpansion(
    knowledge_vector_database=knowledge_vector_database,
    model=model,
    prompt_template=prompt_template,
    question_generator=question_generator
)

# ---------------------------- Question Generaor --------------------------- #

messages_hypothetical = [
    SystemMessage(
        role="system",
        content="Generate 3 hypothetical questions based on the following text. "
                "The results should be formatted as a list, with each question separated by a newline."
    ),
    HumanMessage(
        role="user",
        content="Here is the text: {text}\n"
                "Generate 3 hypothetical questions about the above text."
    ),
]

# Create the ChatPromptTemplate from the messages
prompt_template_hypothetical = ChatPromptTemplate.from_messages(
    [(msg.role, msg.content) for msg in messages_hypothetical]
)

generator = HypotheticalQuestionGenerator(
    model=llm, prompt_template=prompt_template_hypothetical)

# ---------------------------- Schemas --------------------------- #


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        title="Message Role",
        description="Role of the sender of the message, typically 'user' or 'system'.",
        example="user"
    )
    content: str = Field(
        ...,
        title="Message Content",
        description="The actual content of the message.",
        example="What is Conda?"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is Conda?",
            }
        }


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        title="Messages",
        description="A list of messages in the chat conversation.",
        example=[
            {
                "role": "system",
                "content": "Using only the information contained in the context, give a comprehensive answer."
            },
            {
                "role": "user",
                "content": "What is Conda?"
            }
        ]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "Using only the information contained in the context, give a comprehensive answer."
                    },
                    {
                        "role": "user",
                        "content": "What is Conda?"
                    }
                ]
            }
        }


class ChatResponse(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        title="Messages",
        description="The original list of chat messages, including the system's generated response appended as the last message.",
        example=[
            {
                "role": "system",
                "content": "Using only the information contained in the context, give a comprehensive answer."
            },
            {
                "role": "user",
                "content": "What is Conda?"
            },
            {
                "role": "system",
                "content": "Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux."
            }
        ]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "Using only the information contained in the context, give a comprehensive answer."
                    },
                    {
                        "role": "user",
                        "content": "What is Conda?"
                    },
                    {
                        "role": "system",
                        "content": "Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux."
                    }
                ]
            }
        }

# ---------------------------- Endpoints --------------------------- #


@ router.post("/generate-answer", response_model=ChatResponse)
async def generate_answer(request: ChatRequest):
    """Generate an answer to a question and append it as the last message."""
    try:
        # Get the user's question from the last message in the list
        user_message = request.messages[-1].content
        answer = rag_chain.answer_question(user_message)
        request.messages.append(ChatMessage(role="system", content=answer))

        # Return the updated messages list with the generated answer appended
        return ChatResponse(messages=request.messages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryResponse(BaseModel):
    original_query: str = Field(..., description="The original user query.")
    expanded_queries: List[str] = Field(
        ..., description="List of expanded versions of the original query.")


@router.post("/expand-query", response_model=QueryResponse)
async def generate_queries(request: ChatRequest):
    """Generate expanded and hypothetical queries for a given question."""
    try:
        # Get the user's question from the last message in the list
        user_message = request.messages[-1].content

        # Generate expanded and hypothetical queries using RAG-based techniques
        expanded_queries = rag_chain.expand_query(user_message)

        # Return the generated queries
        return QueryResponse(original_query=user_message, expanded_queries=expanded_queries)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define the request and response models
class DocumentRequest(BaseModel):
    document: str = Field(
        default="Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.",
        description="The document to generate questions from."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document": "Conda is an open-source package management system that allows users to install and manage packages and environments."
            }
        }


class QuestionsResponse(BaseModel):
    document: str = Field(
        default="Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.",
        description="The original document provided."
    )
    generated_questions: List[str] = Field(
        default=[
            "What is Conda?",
            "What operating systems does Conda support?",
            "How does Conda manage packages and dependencies?"
        ],
        description="List of generated hypothetical questions."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document": "Conda is an open-source package management system that allows users to install and manage packages and environments.",
                "generated_questions": [
                    "What is Conda?",
                    "How does Conda manage dependencies?",
                    "What platforms are supported by Conda?"
                ]
            }
        }


@router.post("/generate-questions", response_model=QuestionsResponse)
async def generate_questions(request: DocumentRequest):
    """Generate hypothetical questions from the provided document."""
    try:
        # Get the document from the request
        document = request.document

        # Generate hypothetical questions
        questions = generator.generate(document)

        # Return the document and the generated questions
        return QuestionsResponse(document=document, generated_questions=questions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
