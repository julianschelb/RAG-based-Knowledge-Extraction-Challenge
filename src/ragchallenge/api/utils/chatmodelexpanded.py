import os
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


class QuestionAnsweringWithQueryExpansion:
    """Class to perform Question Answering with Query Expansion using Hypothetical Question Generation."""

    def __init__(self, model, prompt_template: ChatPromptTemplate, knowledge_vector_database=None, question_generator=None):
        """
        Initialize the QuestionAnsweringWithQueryExpansion class with an optional knowledge vector database,
        LLM, prompt template, and optional question generator.

        :param model: The language model (LLM) for generating answers.
        :param prompt_template: A LangChain ChatPromptTemplate for answering questions.
        :param knowledge_vector_database: Optional knowledge vector database for retrieval (if provided).
        :param question_generator: Optional question generator for generating alternative queries.
        """
        self.prompt_template = prompt_template
        self.model = model
        self.question_generator = question_generator
        self.retriever = knowledge_vector_database.as_retriever(
        ) if knowledge_vector_database else RunnablePassthrough()
        self.knowledge_vector_database = knowledge_vector_database

        # Define the retrieval chain
        self.retrieval_chain = (
            # {"context": self.retriever, "question": RunnablePassthrough()}
            self.prompt_template
            | self.model
            | StrOutputParser()
        )

    def expand_query(self, question: str) -> List[str]:
        """Generate alternative questions using the question generator."""
        if self.question_generator:
            # Original + expanded queries
            return [question] + self.question_generator.rephrase(question)
        return [question]

    def retrieve_documents(self, questions: List[str], k: int = 1) -> List[str]:
        """Retrieve documents from the vector store for each question."""
        documents = []
        for q in questions:
            retrieved_docs = self.knowledge_vector_database.similarity_search(
                q, k=k)  # Retrieve 1 document per query
            documents.extend([doc.page_content for doc in retrieved_docs])
        return documents

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the LLM, optionally expanding the query and retrieving additional context.

        :param question: The question to answer.
        :return: The generated answer.
        """
        # Expand the query using the hypothetical question generator
        questions = self.expand_query(question)
        # print(f"Cuestions:\n{questions}\n")

        # Retrieve documents for each query (original + expanded)
        context_documents = self.retrieve_documents(questions)

        # Combine the retrieved documents into one context string
        context = "\n".join(context_documents)
        # print(f"Context:\n{context}\n")

        # Invoke the retrieval chain with the combined context and original question
        return self.retrieval_chain.invoke({"context": context, "question": question})


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

    # Assume KNOWLEDGE_VECTOR_DATABASE and QUESTION_GENERATOR are defined (replace with actual instances)
    knowledge_vector_database = None  # Set to actual database if available
    question_generator = None  # Set to actual HypotheticalQuestionGenerator if available

    # Instantiate the QuestionAnsweringWithQueryExpansion class
    qa_with_expansion = QuestionAnsweringWithQueryExpansion(
        knowledge_vector_database=knowledge_vector_database,
        model=model,
        prompt_template=prompt_template,
        question_generator=question_generator
    )

    # Ask a question
    question = "How to start conda?"

    # Get the answer
    answer = qa_with_expansion.answer_question(question)

    # Output the generated answer
    print("\nGenerated Answer:")
    print(answer)
