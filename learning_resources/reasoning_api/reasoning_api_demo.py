import os
import sys
import json
from pydantic import BaseModel, Field, List
from typing import Optional, Literal, Any, List

class Step(BaseModel):
    task: str
    explanation: str
    output: str

class Reasoning(BaseModel):
    thinking: List[Step]
    final_answer: str

class Turn(BaseModel):
    turn_id: int
    user_message: Optional[str] = Field(None, description="The user's message.")
    user_reasoning_turn: List[Reasoning] = Field(default_factory=list, description="Reasoning for this turn.")
    llm_response: Optional[str] = Field(None, description="The LLM's response.")
    llm_reasoning_turn: List[Reasoning] = Field(default_factory=list, description="Reasoning for this turn.")

class Message(BaseModel):
    message_id: int
    conversation: List[Turn] = Field(default_factory=list, description="Alternating user messages and LLM responses.")

class Dialog(BaseModel):
    dialog: List[Message]  = Field(default_factory=list, description="List of conversation message pairs with reasoning.")

    @classmethod
    def create_dialog(cls, n: int, pairs: int):
        """
        Create a dialog with a specified number (n) of message-response pairs, each with 'n' user-LLM message sequences.
        """
        dialog_pairs = []
        for pair_id in range(1, pairs + 1):
            sequence = []
            for message_id in range(1, n + 1):
                sequence.append({
                    "user_message": f"User message {message_id} for pair {pair_id}.",
                    "user__reasoning": f"Reasoning for {message_id} for pair {pair_id}.",
                    "llm_response": f"LLM response {message_id} for pair {pair_id}.",
                    "llm__reasoning": f"Reasoning for {message_id} for pair {pair_id}.",
                })
            dialog_pairs.append(
                Message(
                    message_id=pair_id,
                    conversation=sequence
                )
            )
        return cls(dialog=dialog_pairs)
