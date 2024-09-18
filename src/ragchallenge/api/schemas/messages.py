from typing import List
from pydantic import BaseModel, Field


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

    questions: List[str] = Field([], title="Questions", description="Paraphrased queries", example=[
                                 "What is Conda?", "What is Conda?", "What is Conda?"])

    documents: List[str] = Field([], title="Documents", description="Documents", example=[
        "Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux."])

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


class QueryResponse(BaseModel):
    original_query: str = Field(..., description="The original user query.")
    expanded_queries: List[str] = Field(
        ..., description="List of expanded versions of the original query.")


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
