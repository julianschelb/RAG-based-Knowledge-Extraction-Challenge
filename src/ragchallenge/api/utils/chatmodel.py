import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


class QuestionAnsweringWithRAG:
    """Class to perform Question Answering with optional Retrieval-Augmented Generation (RAG)."""

    def __init__(self, model, prompt_template: ChatPromptTemplate, knowledge_vector_database=None):
        """
        Initialize the QuestionAnsweringWithRAG class with an optional knowledge vector database, LLM, and prompt template.

        :param model: The language model (LLM) for generating answers.
        :param prompt_template: A LangChain ChatPromptTemplate for answering questions.
        :param knowledge_vector_database: Optional knowledge vector database for retrieval (if provided).
        """
        self.prompt_template = prompt_template
        self.model = model
        self.retriever = knowledge_vector_database.as_retriever(
        ) if knowledge_vector_database else RunnablePassthrough()

        self.retrieval_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.model
            | StrOutputParser()
        )

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the LLM and optionally the knowledge vector database.

        :param question: The question to answer.
        :return: The generated answer.
        """
        return self.retrieval_chain.invoke(question)


# Example usage of the class
if __name__ == "__main__":

    # Define the prompt template to generate an answer based on context
    messages = [
        SystemMessage(
            role="system",
            content="""Using the only information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            If the answer cannot be deduced from the context, do not give an answer."""
        ),
        HumanMessage(
            role="user",
            content="""Context:
            {context}
            ---
            Now here is the question you need to answer.

            Question: {question}"""
        ),
    ]

    # Create the ChatPromptTemplate object
    prompt_template = ChatPromptTemplate.from_messages(
        [(msg.role, msg.content) for msg in messages]
    )

    # Example language model (replace with actual LLM)
    repo_id = "HuggingFaceH4/zephyr-7b-beta"  # Model ID from Hugging Face
    task = "text-generation"  # Task type

    # Create an instance of the language model (e.g., HuggingFaceEndpoint or another LLM)
    model = ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ))

    # Assume KNOWLEDGE_VECTOR_DATABASE is already defined, set to None if not using a vector store
    # Replace with actual database or set to None for no retrieval
    knowledge_vector_database = None

    # Instantiate the QuestionAnsweringWithRAG class
    qa_rag = QuestionAnsweringWithRAG(
        knowledge_vector_database=knowledge_vector_database,
        model=model,
        prompt_template=prompt_template,

    )

    # Ask a question
    question = "How to start conda?"

    # Get the answer
    answer = qa_rag.answer_question(question)

    # Output the generated answer
    print("\nGenerated Answer:")
    print(answer)
