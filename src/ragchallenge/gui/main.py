import gradio as gr
import requests

# API endpoint URL
API_URL = "http://localhost:8080/generate-answer"


def prepare_request_payload(message, history):
    """
    Prepare the request payload by extracting the last two history messages and appending the current user message.
    """
    # Get the last two messages from the history
    history_messages = history[-2:] if len(history) >= 2 else history
    history_messages = [
        {"role": "user" if h[0] == "User" else "system", "content": h[1]}
        for h in history_messages
    ]

    # Append the current user message to the history
    history_messages.append({"role": "user", "content": message})

    # Prepare the request payload
    request_data = {
        "messages": history_messages
    }

    return request_data


def get_response_from_api(message, history):
    """
    Sends a user message to the FastAPI endpoint and retrieves the generated answer.
    """
    # Prepare the request payload
    request_data = prepare_request_payload(message, history)

    try:
        # Send a POST request to the FastAPI API and return the answer
        response = requests.post(API_URL, json=request_data)
        response.raise_for_status()  # Check for HTTP errors
        response_data = response.json()

        # Extract the last system message (the answer)
        return response_data["messages"][-1]["content"]

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to reach the API. Details: {e}"


# Gradio Chat Interface
demo = gr.ChatInterface(
    fn=get_response_from_api,
    examples=[
        "What is Conda?",
        "How do I start Conda?",
        "How do I create a new Git branch?",
        "What is the difference between Git fetch and Git pull?",
        "How do I write a regular expression to match an email address?",
        "What is a non-capturing group in regular expressions?"
    ],
    title="RAG - Question Answering Bot",
    multimodal=False,
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
