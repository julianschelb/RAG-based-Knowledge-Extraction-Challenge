# ---------------------------------------------------------------------------
#                            User Request Schema
# ---------------------------------------------------------------------------
# See https://huggingface.co/docs/transformers/main_classes/text_generation

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from enum import Enum

# --------------------------------- Request --------------------------------


class UserRequest(BaseModel):
    prompt: str = Field(
        default="",
        title="Prompt",
        description="Input given to the language model for guiding its generation, providing context and instructions for the desired output or response.",
    )

    max_new_tokens: int = Field(
        default=20,
        title="Max New Tokens",
        description="By setting a value you can control the length of the model's response or output.",
    )

    typical_p: float = Field(
        default=0.2,
        title="Typical Probability",
        description="Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation.",
    )

    temperature: float = Field(
        default=0.6,
        title="Temperature",
        description="The value used to modulate the next token probabilities.",
    )

    num_beams: int = Field(
        default=1,
        title="Number of beams",
        description="Number of beams for beam search. 1 means no beam search",
    )

    num_return_sequences: int = Field(
        default=1,
        title="Number of return sequences",
        description="The number of independently computed returned sequences for each element in the batch.",
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "<|prompter|>Who was Albert Einstein?<|endoftext|><|assistant|>",
                "max_new_tokens": 20,
                "typical_p": 0.2,
                "temperature": 0.6,
                "num_beams": 2,
                "num_return_sequences": 2,
            }
        }


# --------------------------------- Response --------------------------------


class SystemResponse(BaseModel):
    sequences: List = Field(
        default=[],
        title="Responses",
        description="Generated responses from the language model.",
    )

    class Config:
        schema_extra = {
            "example": {
                "sequences": [
                    "<|prompter|>Who was Albert Einstein?<|endoftext|><|assistant|>Albert Einstein was a German physicists. He is widely regarded as one of the most influential scientists",
                    "<|prompter|>Who was Albert Einstein?<|endoftext|><|assistant|>Albert Einstein was a German physicists. He is best known for his theory of relativity and his"
                ]
            }
        }
