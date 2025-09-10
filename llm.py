"""
LLM integration for the ERP system.

This module provides a wrapper for local LLM integration via LM Studio.
It exposes functions for sending prompts and multi-turn messages to the local
LLM and parsing its response. The wrapper connects to LM Studio running on
localhost:1234 by default.
"""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
import os
import json
import requests
from typing import Any, Dict, Iterable, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()




class LLMError(RuntimeError):
    """Custom exception raised when an LLM call fails.
    
    This typically occurs when:
    1. LM Studio is not running
    2. LM Studio local server is not started
    3. The LLM model is not loaded
    4. Network connectivity issues
    """



def get_gemini_client(model: str = "gemini-2.5-flash-lite", temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Initialize and return a ChatGoogleGenerativeAI client."""
    api_key = os.getenv("GOOGLE_API_KEY")  # Ensure the API key is set
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=temperature)


def call_gemini(messages: List[Dict[str, str]]) -> str:
    """Call the Gemini API with the given messages."""
    try:
        # Log the messages for debugging
        logging.debug(f"Messages passed to Gemini API: {messages}")

        # Ensure messages are properly formatted
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
            raise ValueError("Each message must be a dictionary with 'role' and 'content' keys.")

        client = get_gemini_client()
        # Use the invoke method instead of __call__
        response = client.invoke(messages)
        logging.debug(f"Raw response from Gemini API: {response.content}")

        # Check if the response is a BaseMessage object
        if isinstance(response, BaseMessage):
            # Extract content from the BaseMessage object
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                # Handle list content if necessary
                return " ".join(
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in response.content
                )
            else:
                raise ValueError("Unexpected content format in BaseMessage.")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

    except Exception as e:
            logging.error(f"Failed to call Gemini API: {e}")
            raise LLMError(f"Failed to call Gemini API: {e}")


def call_gemini_prompt(prompt: str) -> str:
    """Convenience wrapper for calling LM Studio with a single prompt.

    The prompt is wrapped into a single user message and sent to LM Studio.
    Returns the plain text response.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return call_gemini(messages)
