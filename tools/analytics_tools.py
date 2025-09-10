"""
Analytics and reporting tools for business intelligence.

This module provides text-to-SQL conversion and chart generation
capabilities for the analytics agent.
"""

from __future__ import annotations

import json
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from typing import Any, Dict, List, Optional, Tuple

from llm import call_gemini_prompt
import database


class TextToSQLTool:
    """Converts natural language queries to SQL."""

    @staticmethod
    def generate_sql(
            query: str,
            domain: str = "general",
            schema_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate SQL from natural language query."""

        # Default schema context
        if not schema_context:
            schema_context = """
    Available tables and key columns:
    - customers: id, name, email, phone, created_at
    - orders: id, customer_id, total, status, created_at
    - order_items: id, order_id, product_id, quantity, price
    - products: id, name, price, description
    - invoices: id, customer_id, total_amount, status, issue_date
    - invoice_lines: id, invoice_id, description, quantity, unit_price
    - invoice_orders: invoice_id, order_id
    - payments: id, customer_id, amount, method, received_at
    - payment_allocations: id, payment_id, invoice_id, amount
    - leads: id, customer_name, contact_email, score, status, created_at
    - stock: product_id, quantity, location
    - stock_movements: id, product_id, change_qty, reason, created_at
    - suppliers: id, name, email, phone
    - supplier_products: supplier_id, product_id, lead_time_days, default_cost
    - purchase_orders: id, supplier_id, status, created_at
    - po_items: id, po_id, product_id, quantity, unit_cost
    - po_receipts: id, po_id, product_id, received_qty, received_at
    - tickets: id, customer_id, subject, body, status, created_at
    - ledger_entries: id, entry_date, created_at
    - ledger_lines: id, entry_id, account, debit, credit
    - chart_of_accounts: account, description
    - saved_reports: id, title, sql, created_at
    - conversations: id, user_id, started_at
    - messages: id, conversation_id, sender, content, created_at
    - users: id, name, email, role, created_at
    - customer_kv: customer_id, key, value
    - documents: id, module, path, tags, created_at
    - glossary: term, definition, module
    - approvals: id, module, payload_json, status, requested_by, decided_at
    - tool_calls: id, agent, tool_name, input_json, output_json, created_at
    """

        # Detect if this is a metric calculation query
        is_metric_calculation = "metric" in query.lower() or "calculate" in query.lower() or "margin" in query.lower()
        metric_guidance = ""

        if is_metric_calculation:
            metric_guidance = """
    For business metric calculations:
    1. DO NOT use Common Table Expressions (CTEs) or WITH clauses
    2. Use simple subqueries or direct joins instead
    3. Break complex calculations into simpler steps
    4. Never Add comments
    5. Include proper NULL handling with COALESCE or IFNULL
    6. Use explicit type casting when dividing (CAST or * 1.0)
    7. Always handle division by zero with CASE statements
    """

        prompt = f"""
    You are an expert SQLite query generator. Convert this natural language query to SQL:

    Query: "{query}"
    Domain: {domain}

    Database Schema:
    {schema_context}

    Requirements:
    - Generate ONLY basic SQLite-compatible SQL
    - Prefer simple, direct queries without advanced features
    - DO NOT use Common Table Expressions (WITH clauses)
    - DO NOT use window functions (PARTITION BY, OVER)
    - Use standard joins (INNER JOIN, LEFT JOIN) when needed
    - Include only columns that exist in the tables
    - Add WHERE clauses to filter appropriately
    - For calculations, use explicit casting (value * 1.0) for division
    - Use CASE statements to handle division by zero

    {metric_guidance}

    Respond with JSON:
    {{
        "sql": "SELECT ...",
        "explanation": "This query retrieves...",
        "confidence": "high",
        "tables_used": ["customers", "orders"],
        "potential_issues": ["None" or list of issues]
    }}

    Confidence levels: low, medium, high
    """

        try:
            response = call_gemini_prompt(prompt)
            logging.debug(f"responseL: {response}")
            # Clean the response by removing markdown code block delimiters
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]

            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()
            result = json.loads(cleaned_response)

            # Validate SQL is present
            if not result.get("sql"):
                raise ValueError("No SQL generated")

            # Basic validation
            sql = result["sql"]
            logging.debug(f"sql1 here: {sql}")
            sql = ' '.join(sql.replace('\n', ' ').split()).strip().strip("`")
            logging.debug(f"sql2 here: {sql}")
            result["sql"] = sql
            sql = result["sql"]

            if "WITH " in sql.upper():
                result["potential_issues"] = result.get("potential_issues", []) + [
                    "Query uses CTE which may not be supported"]
                result["confidence"] = "low"

            # Check for other complex features
            if any(x in sql.upper() for x in ["PARTITION BY", "OVER(", "OVER ("]):
                result["potential_issues"] = result.get("potential_issues", []) + [
                    "Query uses window functions not supported in SQLite"]
                result["confidence"] = "low"

            return result

        except Exception as e:
            logging.debug(f"Warning: SQL generation failed: {e}")
            return {
                "sql": "SELECT 'Error generating SQL' as error_message",
                "explanation": f"Failed to generate SQL: {str(e)}",
                "confidence": "low",
                "tables_used": [],
                "potential_issues": ["SQL generation failed"]
            }
    
    @staticmethod
    def execute_generated_sql(query: str, domain: str = "general") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate and execute SQL from natural language."""
        sql_result = TextToSQLTool.generate_sql(query, domain)
        
        try:
            # Execute the SQL
            sql_query = sql_result["sql"]
            data = database.query(sql_query, None)
            return data or [], sql_result
            
        except Exception as e:
            print(f"Warning: SQL execution failed: {e}")
            return [], {
                **sql_result,
                "execution_error": str(e),
                "confidence": "low"
            }


class AnalyticsReportingTool:
    """Generates chart specifications and formatted reports."""
    
    @staticmethod
    def generate_chart_spec(
        data: List[Dict[str, Any]],
        chart_type: str = "auto",
        title: str = "",
        query_context: str = ""
    ) -> Dict[str, Any]:
        """Generate chart specification for data visualization."""
        
        if not data:
            return {
                "type": "message",
                "title": title or "No Data",
                "message": "No data available for visualization",
                "data": []
            }
        
        # Analyze data structure
        sample_row = data[0]
        columns = list(sample_row.keys())
        
        # Prepare data summary for LLM
        data_summary = f"Data has {len(data)} rows with columns: {', '.join(columns)}\n"
        data_summary += f"Sample row: {sample_row}\n"
        
        if len(data) > 1:
            data_summary += f"Last row: {data[-1]}\n"
        
        prompt = f"""
You are a data visualization expert. Analyze this data and create a chart specification:

Query Context: {query_context}
{data_summary}

Chart Type Preference: {chart_type}

Create a chart specification that best represents this data. Consider:
- Data types (numerical, categorical, dates)
- Number of data points
- Relationships in the data
- Best visualization type for the query context

Respond with JSON:
{{
    "type": "bar|line|pie|table|scatter",
    "title": "Chart Title",
    "x_axis": "column_name",
    "y_axis": "column_name", 
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "data": [prepared data for chart],
    "summary": "Brief description of what the chart shows",
    "insights": ["Key insight 1", "Key insight 2"]
}}

Chart Types:
- bar: categorical data, comparisons
- line: time series, trends
- pie: proportions, percentages  
- table: detailed data, multiple columns
- scatter: correlations, relationships
"""
        
        try:
            response = call_gemini_prompt(prompt)
            result = json.loads(response.strip())
            
            # Ensure we have the original data
            result["raw_data"] = data
            result["row_count"] = len(data)
            
            # If no title provided, use the generated one
            if not title and result.get("title"):
                result["title"] = result["title"]
            elif title:
                result["title"] = title
            
            return result
            
        except Exception as e:
            print(f"Warning: Chart generation failed: {e}")
            # Fallback to table format
            return {
                "type": "table",
                "title": title or "Data Table",
                "data": data,
                "raw_data": data,
                "row_count": len(data),
                "summary": f"Table view of {len(data)} records",
                "insights": ["Data visualization failed, showing raw table"]
            }
    
    @staticmethod
    def create_report(
        query: str,
        domain: str = "general",
        include_chart: bool = True
    ) -> Dict[str, Any]:
        """Create a complete analytics report with data and visualization."""
        
        # Generate and execute SQL
        data, sql_info = TextToSQLTool.execute_generated_sql(query, domain)
        
        # Create chart if requested and data is available
        chart_spec = None
        if include_chart and data:
            chart_spec = AnalyticsReportingTool.generate_chart_spec(
                data, 
                title=f"Analysis: {query}",
                query_context=query
            )
        
        # Generate narrative summary
        narrative = AnalyticsReportingTool._generate_narrative(query, data, sql_info)
        
        return {
            "query": query,
            "sql_info": sql_info,
            "data": data,
            "chart": chart_spec,
            "narrative": narrative,
            "timestamp": database.query("SELECT datetime('now') as now", None)[0]["now"]
        }
    
    @staticmethod
    def _generate_narrative(query: str, data: List[Dict[str, Any]], sql_info: Dict[str, Any]) -> str:
        """Generate a narrative summary of the analysis."""
        
        if not data:
            return f"No data found for the query: '{query}'. This could indicate that there are no records matching the criteria or the query needs to be refined."
        
        data_summary = f"Found {len(data)} records. "
        
        # Add basic insights based on data
        if len(data) == 1:
            data_summary += "Here are the details for the single result found."
        elif len(data) < 10:
            data_summary += "Here is a summary of the key findings."
        else:
            data_summary += "Here are the key insights from the analysis."
        
        # Add SQL confidence info
        confidence = sql_info.get("confidence", "medium")
        if confidence == "high":
            data_summary += " The analysis was performed with high confidence in the data accuracy."
        elif confidence == "low":
            data_summary += " Please note that this analysis has lower confidence due to query complexity or data limitations."
        
        return data_summary


# Convenience functions
def text_to_sql(query: str, domain: str = "general") -> str:
    """Quick text-to-SQL conversion returning just the SQL."""
    result = TextToSQLTool.generate_sql(query, domain)
    return result.get("sql", "")


def quick_analysis(query: str, domain: str = "general") -> Dict[str, Any]:
    """Quick analysis with chart generation."""
    return AnalyticsReportingTool.create_report(query, domain, include_chart=True)


def table_only_analysis(query: str, domain: str = "general") -> List[Dict[str, Any]]:
    """Quick analysis returning just the data table."""
    data, _ = TextToSQLTool.execute_generated_sql(query, domain)
    return data
