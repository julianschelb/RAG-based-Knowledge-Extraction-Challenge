
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from ragchallenge.api.llm import LLM
from ragchallenge.api.interfaces.paraphraser import QueryParaphraser

# ---------------------------- Load Paraphraser --------------------------- #

messages = [
    SystemMessage(
        role="system",
        content="Generate 3 paraphrased versions of the question provided by the user. "
                "The results should be formatted as a list, each paraphrase separated by a newline."
    ),
    HumanMessage(
        role="user",
        content="Generate 3 paraphrased versions of the following question: {question}"
    ),
]

# Create the ChatPromptTemplate from the messages
prompt_template_paraphrase = ChatPromptTemplate.from_messages(
    [(msg.role, msg.content) for msg in messages]
)

# Create an instance of QueryParaphraser
PARAPHRASER = QueryParaphraser(
    model=LLM, prompt_template=prompt_template_paraphrase)
