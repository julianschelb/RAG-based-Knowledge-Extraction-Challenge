import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


class QueryParaphraser:
    """Class to generate paraphrased versions of a given question."""

    def __init__(self, model, prompt_template: ChatPromptTemplate):
        """
        Initialize the QueryParaphraser class with the LLM and prompt template.

        :param model: The Hugging Face model to use.
        :param prompt_template: A LangChain ChatPromptTemplate to generate paraphrased queries.
        """
        # Store the prompt template and the LLM model
        self.prompt_template = prompt_template
        self.llm = model
        self.paraphrasing_chain = self.prompt_template | self.llm | self.parse_output

    @staticmethod
    def parse_output(result):
        """Parses the result and splits it into a list of paraphrased queries based on newlines."""
        result = result.content
        return [line.strip() for line in result.split('\n') if line.strip()]

    def rephrase(self, question: str) -> list:
        """
        Generate paraphrased versions of the given question.

        :param question: The input question to paraphrase.
        :return: A list of paraphrased queries.
        """
        # Prepare the input dictionary for the chain
        input_dict = {"question": question}

        # Invoke the chain to get the paraphrased questions
        paraphrased_questions = self.paraphrasing_chain.invoke(input_dict)

        return paraphrased_questions


# Example usage of the class
if __name__ == "__main__":
    # Define the prompt template to paraphrase the question
    messages = [
        SystemMessage(
            role="system",
            content="Generate 5 paraphrased versions of the question provided by the user. "
                    "The results should be formatted as a list, each paraphrase separated by a newline."
        ),
        HumanMessage(
            role="user",
            content="Generate 5 paraphrased versions of the following question: {question}"
        ),
    ]

    # Create the ChatPromptTemplate from the messages
    prompt_template_paraphrase = ChatPromptTemplate.from_messages(
        [(msg.role, msg.content) for msg in messages]
    )

    # Define the Hugging Face model to use for paraphrasing
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

    # Create an instance of QueryParaphraser
    paraphraser = QueryParaphraser(
        model=llm, prompt_template=prompt_template_paraphrase)

    # Example input question
    question = "What is conda in Python?"

    # Generate paraphrased versions of the input question
    paraphrased_questions = paraphraser.rephrase(question)

    # Output the generated paraphrased questions
    print("\nGenerated Paraphrased Questions:")
    for idx, paraphrased in enumerate(paraphrased_questions, 1):
        print(f"{idx}. {paraphrased}")
