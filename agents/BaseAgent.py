# agents/base_agent.py
import json
from typing import Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool
from llm import call_gemini_prompt, LLMError

class BaseAgent:
    """Base class for all agents to enforce a consistent model context protocol."""

    def __init__(self, domain: str, tools: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the base agent with common tools."""
        self.domain = domain
        self.sql = SQLTool(domain)
        self.rag = VectorRAGTool("main_documents")  # General RAG tool
        self.tools = tools or {}

    def get_default_intent_prompt(self, query: str) -> str:
        """Return a default system prompt for intent analysis."""
        INTENT_PROMPT = ChatPromptTemplate.from_template("""You are an assistant for the {domain} domain.
        Analyze the user's query and classify the type of action needed.
        query: {query}
         Respond with ONLY a JSON object:
        {
            "action_type": "retrieve_data|create_record|update_record|delete_record|other",
            "filters": ["list of filters or conditions"],
            "context": "brief description of what the user wants"
        }
        """)
        messages = INTENT_PROMPT.format_messages(domain=self.domain, query=query)
        return messages[0].content

    def _analyze_intent(self, query: str) -> dict:
        """Use LLM to analyze user intent and classify the type of action needed."""
        prompt = self.get_default_intent_prompt(query)
        try:
            response = call_gemini_prompt(prompt)
            # Clean up and parse the JSON response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            return json.loads(response)
        except Exception as e:
            # Fallback intent
            return {
                "action_type": "other",
                "filters": [],
                "context": "Unable to determine intent"
            }

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        """Handle a user query. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `handle_query` method.")

    def _summarise(self, raw_response: str, query: str) -> str:
        """Summarize raw data using the LLM."""
        prompt = (
            f"You are an assistant for the {self.domain} domain summarizing data for a user. "
            "Given the following raw information and the user's query, "
            "compose a concise and helpful reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _fallback_response(self, query: str) -> str:
        """Provide a fallback response if the query cannot be handled."""
        return f"I'm sorry, I couldn't understand your {self.domain} request."