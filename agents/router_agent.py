"""
Router agent implementation.

The RouterAgent is responsible for determining which domain agent should
handle a given user query.  It does this by calling the configured LLM to
classify the query into one of the recognised domains.  After selecting
the domain, it delegates processing to the appropriate agent.  The router
also records conversations, messages and tool invocations in the database.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import database
from llm import call_gemini, LLMError
from tools.sql_tool import SQLTool


class RouterAgent:
    """Entry point for handling conversational queries."""

    def __init__(self, agents: Dict[str, object]) -> None:
        """Initialise the router with a mapping of domain names to agent instances."""
        self.agents = agents
        # Tools used by the router to log calls and approvals
        self.router_sql = SQLTool("router")

    def _create_conversation(self, user_id: int) -> int:
        """Create a new conversation row and return its ID."""
        sql = "INSERT INTO conversations (user_id, started_at) VALUES (?, datetime('now'))"
        return self.router_sql.write(sql, (user_id,))

    def _add_message(self, conversation_id: int, sender: str, content: str) -> int:
        """Insert a message into the messages table and return its ID."""
        sql = (
            "INSERT INTO messages (conversation_id, sender, content, created_at) "
            "VALUES (?, ?, ?, datetime('now'))"
        )
        return self.router_sql.write(sql, (conversation_id, sender, content))

    def _log_tool_call(self, agent: str, tool_name: str, input_json: Dict, output_json: Dict) -> int:
        """Record a tool invocation in the tool_calls table."""
        sql = (
            "INSERT INTO tool_calls (agent, tool_name, input_json, output_json, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))"
        )
        return self.router_sql.write(
            sql,
            (
                agent,
                tool_name,
                json.dumps(input_json, default=str),
                json.dumps(output_json, default=str),
            ),
        )

    def classify_domain(self, query: str) -> str:
        """Call the LLM to classify the user's query into a domain.

        This method uses a simple prompt to instruct the model to respond with
        one of the recognised domain names: "Sales", "Finance", "Inventory",
        or "Analytics".  It returns the lowercase domain string.  Any
        unexpected response will result in a `ValueError`.
        """
        system_prompt = (
            "You are a domain classifier for a modular ERP system.  "
            "Classify user requests into exactly one of these domains:\n"
            "- Sales: customer management, leads, orders, products, CRM activities\n"
            "- Finance: invoices, payments, accounting, billing\n"
            "- Inventory: stock levels, suppliers, purchase orders, warehouse management, listing and retrieving product information\n"
            "- Analytics: reports, revenue analysis, profit analysis, data insights, trends, statistics\n"
            "Respond with exactly one domain name: Sales, Finance, Inventory, or Analytics."
        )
        messages = [
            {"role": "user", "content": [{"text": f"{system_prompt}\n\nUser request: {query}"}]},
        ]
        try:
            response = call_gemini(messages)
        except LLMError as e:
            raise
        # Normalise the response: take the first word, strip punctuation and whitespace
        domain_raw = response.strip().split()[0]
        # Remove punctuation and convert to lowercase
        import string
        domain = domain_raw.lower().strip(string.punctuation)
        if domain not in {"sales", "finance", "inventory", "analytics"}:
            raise ValueError(f"Unexpected domain classification: {response}")
        return domain

    def handle_chat(self, query: str, conversation_id: Optional[int] = None, user_id: int = 1) -> Dict[str, str]:
        """Process a user message.

        Inserts the message into the conversation history, determines the domain
        using the LLM, delegates handling to the domain agent and records the
        response.  Returns a dictionary with the conversation ID and the agent's
        textual response.
        """
        # Ensure we have a conversation
        if conversation_id is None:
            conversation_id = self._create_conversation(user_id)
        # Record the user message
        self._add_message(conversation_id, sender="user", content=query)
        # Determine which agent should handle the request
        domain: str
        try:
            domain = self.classify_domain(query)
        except Exception as e:
            # Record error as tool call
            self._log_tool_call(
                "router",
                "classify_domain",
                {"query": query},
                {"error": str(e)},
            )
            raise
        agent = self.agents.get(domain)
        if agent is None:
            raise RuntimeError(f"No agent registered for domain '{domain}'")
        # Delegate to the agent
        try:
            response_text = agent.handle_query(query, conversation_id)
        except Exception as e:
            # Log any agent error
            self._log_tool_call(
                domain,
                "handle_query",
                {"query": query},
                {"error": str(e)},
            )
            raise
        # Record the agent's reply as a message
        self._add_message(conversation_id, sender=domain, content=response_text)
        # Log the call
        self._log_tool_call(
            domain,
            "handle_query",
            {"query": query},
            {"response": response_text},
        )
        return {"conversation_id": conversation_id, "response": response_text}
