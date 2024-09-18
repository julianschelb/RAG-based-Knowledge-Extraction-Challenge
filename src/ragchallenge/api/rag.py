from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ragchallenge.api.database import DATABASE
from ragchallenge.api.paraphraser import PARAPHRASER
from ragchallenge.api.llm import LLM
from ragchallenge.api.utils.chatmodelexpanded import QuestionAnsweringWithQueryExpansion

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

# Instantiate the QuestionAnsweringWithRAG class once
RAG_MODEL = QuestionAnsweringWithQueryExpansion(
    knowledge_vector_database=DATABASE.vector_store,
    prompt_template=prompt_template,
    question_generator=PARAPHRASER,
    model=LLM)
