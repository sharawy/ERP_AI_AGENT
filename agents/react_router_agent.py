"""
LangChain ReAct agent router for ERP system.

This module wraps each domain agent as a LangChain Tool and uses a ReAct agent to route queries.
"""
import os

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.sales_agent import SalesAgent
from agents.finance_agent import FinanceAgent
from agents.inventory_agent import InventoryAgent
from agents.analytics_agent import AnalyticsAgent
from llm import get_gemini_client


class LangchainRouter:
    def __init__(self):
        self.sales_agent = SalesAgent()
        self.finance_agent = FinanceAgent()
        self.inventory_agent = InventoryAgent()
        self.analytics_agent = AnalyticsAgent()

        self.tools = [
                Tool(
                    name="Sales",
                    func=self.sales_agent.handle_query,
                    description=(
                        "Handles customer management, leads, orders, products, and CRM activities. "
                        "Use this tool for any queries related to sales operations, customer data, or product information."
                    ),
                ),
                Tool(
                    name="Finance",
                    func=self.finance_agent.handle_query,
                    description=(
                        "Manages invoices, payments, accounting, and billing. "
                        "Use this tool for financial workflows, such as creating invoices, listing payments, or answering finance-related queries."
                    ),
                ),
                Tool(
                    name="Inventory",
                    func=self.inventory_agent.handle_query,
                    description=(
                        "Handles stock levels, suppliers, purchase orders, and warehouse management. "
                        "Use this tool for inventory-related queries, such as restocking, retrieving product information, or managing suppliers."
                    ),
                ),
                Tool(
                    name="Analytics",
                    func=self.analytics_agent.handle_query,
                    description=(
                        "Provides reports, revenue analysis, profit analysis, data insights, trends, and statistics. "
                        "Use this tool for analytics-related queries, such as generating reports or analyzing business metrics."
                    ),
                ),
        ]
        api_key = os.getenv("GOOGLE_API_KEY")  # Ensure the API key is set
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        self.llm = get_gemini_client()
        self.agent_executor = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, query: str) -> str:
        return self.agent_executor.run(query)