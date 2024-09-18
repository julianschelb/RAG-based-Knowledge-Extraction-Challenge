
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ragchallenge.api.llm import LLM
from ragchallenge.api.interfaces.generator import HypotheticalQuestionGenerator

# ---------------------------- Load Paraphraser --------------------------- #

# Prompt template to generate hypothetical questions
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

QUESTION_GENERATOR = HypotheticalQuestionGenerator(
    model=LLM, prompt_template=prompt_template_hypothetical)
