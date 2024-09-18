from fastapi import APIRouter, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


from ragchallenge.api.interfaces.generator import HypotheticalQuestionGenerator
from ragchallenge.api.schemas.messages import DocumentRequest, QuestionsResponse

router = APIRouter(responses={404: {"description": "Not Found"}})

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

# Load your language model (replace with actual Hugging Face model)
repo_id = "HuggingFaceH4/zephyr-7b-beta"
task = "text-generation"

# Create the Hugging Face Endpoint using the specified parameters
endpoint = HuggingFaceEndpoint(
    repo_id=repo_id,
    task=task,
    **generation_params,
)

# Return the LangChain HuggingFacePipeline object with the endpoint
llm = ChatHuggingFace(llm=endpoint)

# ---------------------------- Load Question Generaor --------------------------- #

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


# ---------------------------- Endpoints --------------------------- #

@router.post("/generate-questions", response_model=QuestionsResponse)
async def generate_questions(request: DocumentRequest):
    """Generate hypothetical questions from the provided document. Is used to enhance retrieval performance."""
    try:
        # Get the document from the request
        document = request.document

        # Generate hypothetical questions
        questions = generator.generate(document)

        # Return the document and the generated questions
        return QuestionsResponse(document=document, generated_questions=questions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
