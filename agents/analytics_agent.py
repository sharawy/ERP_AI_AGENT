"""
Analytics & Reporting agent.

This agent answers quantitative and reasoning‑based questions by executing
SQL queries and summarising the results.  It supports listing saved
reports and running a specific saved report by its ID.  More complex
analytics can be added by extending the keyword heuristics or by
integrating a text‑to‑SQL model via the LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from llm import call_gemini_prompt, LLMError
from tools.analytics_tools import  TextToSQLTool


from agents.BaseAgent import BaseAgent
from tools.vector_rag_tool import VectorRAGTool, DefinitionRAGTool
from tools.saved_reports import SavedReportsManager
from tools.audit_logger import audit_agent_method
import re
from typing import Optional, Dict, Any, List


class AnalyticsAgent(BaseAgent):
    """Analytics & Reporting agent that answers quantitative questions by executing SQL queries."""

    def __init__(self) -> None:
        """Initialize the analytics agent with domain-specific tools."""
        super().__init__(domain="analytics")

        # Domain-specific RAG tools
        try:
            self.definition_rag = DefinitionRAGTool()
            self.metrics_rag = VectorRAGTool("metrics_documents")
        except Exception as e:
            print(f"Warning: Could not initialize specialized RAG tools: {e}")
            self.definition_rag = None
            self.metrics_rag = None

    def get_default_intent_prompt(self, query: str) -> str:
        """Return a specialized system prompt for analytics intent analysis."""
        analytics_prompt = ChatPromptTemplate.from_template("""You are an assistant for the {domain} domain.
    Analyze the user's query and classify the type of action needed for analytics and reporting.
    query: {query}
    Respond with ONLY a JSON object:
    {{
        "action_type": "run_report|list_reports|revenue_analysis|customer_analysis|inventory_analysis|metric_definition|metric_calculation|custom_query",
        "filters": ["list of filters or time periods mentioned"],
        "context": "brief description of what the user wants",
        "entities": {{
            "report_id": "extracted report ID if present",
            "metric": "specific metric mentioned (CLV, CAC, MRR, Churn Rate, AOV, etc.)",
            "time_period": "time period for analysis (daily, monthly, quarterly, etc.)",
            "product_id": "specific product ID if mentioned",
            "customer_id": "specific customer ID if mentioned"
        }}
    }}
    Additional Notes:
    - Use 'run_report' when the user wants to run a specific saved report by ID
    - Use 'list_reports' when the user wants to see available reports
    - Use 'revenue_analysis' for questions about sales performance, revenue trends, or profitability
    - Use 'metric_definition' when the user asks for the meaning or definition of a business metric
    - Use 'metric_calculation' when the user wants to calculate a specific business metric (use the formula from definitions)
    - Use 'custom_query' for other analytical questions requiring SQL generation""")
        messages = analytics_prompt.format_messages(domain=self.domain, query=query)
        return messages[0].content

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        """Process an analytics query using intent-based routing."""
        try:
            # Analyze user intent
            query += str(self.definition_rag.search_definitions(query))
            intent = self._analyze_intent(query)
            action_type = intent.get("action_type", "custom_query")
            entities = intent.get("entities", {})

            # Route based on intent action_type
            if action_type == "list_reports":
                return self._list_saved_reports(query)
            elif action_type == "run_report":
                report_id = entities.get("report_id")
                if not report_id:
                    # Extract report ID using regex as fallback
                    m = re.search(r"report\s+(\d+)", query.lower())
                    report_id = int(m.group(1)) if m else None

                if report_id:
                    return self._run_saved_report(int(report_id), query)
                return self._summarise("No report ID specified. Please provide a report ID.", query)

            elif action_type == "metric_definition":
                metric = entities.get("metric", "")
                if self.definition_rag:
                    rag_results = self.definition_rag.search_definitions(query)
                    if rag_results:
                        return self._summarise(f"Metric definition: {rag_results}", query)

            elif action_type in ("metric_calculation", "customer_analysis"):
                metric = entities.get("metric", "")
                # First search for the metric definition
                if self.definition_rag:
                    rag_results = self.definition_rag.search_definitions(metric if metric else query)
                    if not rag_results:
                        return self._summarise(f"I couldn't find a definition for the {metric} metric.", query)

                    # Now generate SQL based on the definition
                    calculation_prompt = f"""
                    Based on this definition: {rag_results}

                    Generate SQL query to calculate the {metric} metric.
                    Use the tables available in the analytics domain.
                    """
                    try:
                        results = []
                        sql_result = TextToSQLTool.generate_sql(calculation_prompt+ " " + query, "analytics")
                        if sql_result and sql_result.get("sql").strip().strip("`"):
                            sql_query = sql_result["sql"]
                            rows = self.sql.read(sql_query)
                            results.append({"sql": sql_query, "results": rows})
                            return self._summarise(str(results), query)
                    except Exception as e:
                        print(f"Text-to-SQL error: {e}")
                        return self._summarise(
                            f"I found the definition for {metric}, but couldn't generate a calculation: {rag_results}",
                            query)

            elif action_type == "revenue_analysis":
                time_period = entities.get("time_period", "").lower()
                if "day" in time_period:
                    return self._analyze_revenue_by_day(query)
                elif "month" in time_period:
                    return self._analyze_revenue_by_month(query)
                else:
                    return self._analyze_total_revenue(query)

            elif action_type == "custom_query":
                # Generate SQL using text-to-SQL
                try:
                    sql_queries = TextToSQLTool.generate_sql(query, "analytics")
                    if sql_queries:
                        results = []
                        for sql in sql_queries:
                            rows = self.sql.read(sql)
                            results.append({"sql": sql, "results": rows})
                        return self._summarise(str(results), query)
                except Exception as e:
                    print(f"Text-to-SQL error: {e}")

            # Fallback RAG search
            if self.metrics_rag:
                rag_results = self.metrics_rag.search(query)
                if rag_results:
                    return self._summarise(f"Metrics information: {rag_results}", query)

            rag_results = self.rag.search(query)
            if rag_results:
                return self._summarise(f"Information found: {rag_results}", query)

            return self._summarise(
                "I couldn't find specific analytics information for your query. Could you please rephrase or provide more details?",
                query)

        except Exception as e:
            print(f"Error processing query: {e}")
            return self._summarise(f"Error processing analytics query: {str(e)}", query)

    # Keep existing analysis methods
    def _list_saved_reports(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, title FROM saved_reports ORDER BY created_at DESC"
        )
        return self._summarise(str(rows), query)

    @audit_agent_method("analytics")
    def _run_saved_report(self, report_id: int, query: str) -> str:
        """Run a saved report by ID."""
        try:
            report_result = SavedReportsManager.execute_report(report_id)
            if report_result.get("error"):
                return self._summarise(report_result["error"], query)

            response = f"Report: {report_result['report_name']}\n"
            response += f"Description: {report_result['description']}\n\n"
            response += report_result["narrative"]

            if report_result.get("chart"):
                chart_info = report_result["chart"]
                response += f"\n\nVisualization: {chart_info['type']} chart"

            return self._summarise(response, query)
        except Exception as e:
            return self._summarise(f"Failed to run report {report_id}: {str(e)}", query)

    def _analyze_revenue_by_day(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT DATE(created_at) as date, SUM(total) as daily_revenue, COUNT(*) as order_count "
            "FROM orders WHERE status != 'cancelled' "
            "GROUP BY DATE(created_at) "
            "ORDER BY daily_revenue DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _analyze_revenue_by_month(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT strftime('%Y-%m', created_at) as month, SUM(total) as monthly_revenue, COUNT(*) as order_count "
            "FROM orders WHERE status != 'cancelled' "
            "GROUP BY strftime('%Y-%m', created_at) "
            "ORDER BY monthly_revenue DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _analyze_total_revenue(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT "
            "SUM(CASE WHEN status != 'cancelled' THEN total ELSE 0 END) as total_revenue, "
            "COUNT(CASE WHEN status != 'cancelled' THEN 1 END) as successful_orders, "
            "COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders, "
            "AVG(CASE WHEN status != 'cancelled' THEN total ELSE NULL END) as avg_order_value "
            "FROM orders"
        )
        logging.debug(f"total_revenue rows: {rows}")
        return self._summarise(str(rows), query)

    def _summarise(self, raw_response: str, query: str) -> str:
        prompt = (
            "You are an analytics assistant summarising data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and insightful reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)