import os
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


class HypotheticalQuestionGenerator:
    """ Class to generate hypothetical questions based on a given document text. """

    def __init__(self, model, prompt_template: ChatPromptTemplate):
        """
        Initialize the HypotheticalQuestionGenerator class with the LLM and prompt template.

        :param model: The Hugging Face model to use.
        :param prompt_template: A LangChain ChatPromptTemplate to generate hypothetical questions.
        """
        # Store the prompt template and the LLM model
        self.prompt_template = prompt_template
        self.llm = model
        self.hypothetical_question_chain = self.prompt_template | self.llm | self.parse_output

    @staticmethod
    def parse_output(result):
        """Parses the result and splits it into a list of questions based on newlines."""
        result = result.content
        return [line.strip() for line in result.split('\n') if line.strip()]

    def generate(self, document: str) -> list:
        """
        Generate hypothetical questions based on the provided document text.

        :param document: The input document text.
        :return: A list of hypothetical questions.
        """
        # Prepare the input dictionary for the chain
        input_dict = {"text": document}

        # Invoke the chain to get the hypothetical questions
        hypothetical_questions = self.hypothetical_question_chain.invoke(
            input_dict)

        return hypothetical_questions

    def expand_query(self, question: str) -> List[str]:
        """Generate alternative questions using the question generator."""
        return [question] + self.generate(question)


# Example usage of the class
if __name__ == "__main__":
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
    )

    # Return the LangChain HuggingFacePipeline object with the endpoint
    llm = ChatHuggingFace(llm=endpoint)

    generator = HypotheticalQuestionGenerator(
        model=llm, prompt_template=prompt_template_hypothetical)

    # Example text input
    document = "Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs, and updates packages and their dependencies."

    # Generate hypothetical questions
    questions = generator.generate(document)

    # Output the generated hypothetical questions
    print("\nGenerated Hypothetical Questions:")
    for idx, question in enumerate(questions, 1):
        print(f"{idx}. {question}")
