"""
Sales & CRM agent.

This agent handles operations related to customers, leads, orders and
support tickets.  It exposes a `handle_query()` method that interprets
user requests, executes SQL commands via the `SQLTool`, performs simple
retrieval over the `documents` table via the `RAGTool` and calls the LLM
to summarise results.  The classification of high‑level intent (Sales vs
Finance etc.) is performed by the RouterAgent; within the Sales domain we
use keyword heuristics to map queries to concrete actions.  You can
replace these heuristics with additional LLM calls if desired.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from agents.BaseAgent import BaseAgent
from llm import call_gemini_prompt, LLMError
from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool, SalesRAGTool
from tools.ml_tools import LeadScoringTool



class SalesAgent(BaseAgent):
    def __init__(self) -> None:
        self.sql = SQLTool("sales")
        self.rag = VectorRAGTool("main_documents")  # General RAG
        self.domain = "sales"
        # Domain-specific RAG tools as per requirements
        try:
            self.sales_rag = SalesRAGTool()
        except Exception as e:
            print(f"Warning: Could not initialize SalesRAGTool: {e}")
            self.sales_rag = None
    def get_default_intent_prompt(self, query: str) -> str:
        intent_prompt = ChatPromptTemplate.from_template(
        """
        You are a Sales & CRM agent. Analyze this user query and determine the action needed.
        
        Query: {query}
        
        You handle: customers, leads, orders, support tickets, products
        
        Classify the query into one of these action types:
        1. "new_lead" - creating a new sales lead
        2. "new_order" - creating a new order
        3. "support_ticket" - handling customer support
        4. "retrieve_customers" - getting customer information
        5. "retrieve_orders" - getting order information  
        6. "retrieve_leads" - getting lead information
        7. "retrieve_products" - getting product information
        8. "rag_query" - asking about procedures, policies, guidelines, definitions, "how to", "what is", "what are"
        9. "other" - anything else
        
        Key indicators for rag_query:
        - Questions about procedures, policies, guidelines
        - "What are our..." / "What is our..." / "How do we..."
        - Questions about qualification, standards, procedures
        - Definitional questions
        
        Also determine any specific filters or conditions mentioned.
        
        Respond with ONLY a JSON object:
        {{
            "action_type": "new_lead|new_order|support_ticket|retrieve_customers|retrieve_orders|retrieve_leads|retrieve_products|rag_query|other",
            "filters": ["cancelled", "recent", "high_value", etc.],
            "context": "brief description of what user wants"
        }}
        """)
        messages = intent_prompt.format_messages(domain=self.domain, query=query)
        return messages[0].content

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        """Intelligently interpret and respond to user queries within the Sales domain.

        This method uses the LLM to understand the user's intent, generates appropriate
        SQL queries, executes them, and provides intelligent analysis of the results.
        """
        try:
            # Step 1: Understand what the user wants
            intent = self._analyze_intent(query)

            # Step 2: Execute the appropriate action based on LLM classification
            action_type = intent.get("action_type", "other")

            if action_type.startswith("retrieve_"):
                return self._handle_retrieval_action(action_type, intent, query)
            elif action_type in ["new_lead", "new_order", "support_ticket"]:
                return self._handle_creation_action(action_type, intent, query)
            elif action_type == "rag_query":
                # Handle RAG queries for procedures, policies, definitions
                return self._handle_rag_query(query)
            else:
                # Fallback: attempt RAG search on documents
                rag_results = self.rag.search(query)
                if rag_results:
                    return self._summarise(
                        f"RAG search results for '{query}': {rag_results}", query
                    )
                return "I'm sorry, I couldn't understand your sales request."
        except Exception:
            # Re-raise so the router can log the error
            raise


    def _handle_retrieval_action(self, action_type: str, intent: dict, original_query: str) -> str:
        """Handle data retrieval actions by generating SQL and analyzing results."""
        # Step 1: Generate SQL query using LLM
        sql_query = self._generate_sql_for_action(action_type, intent, original_query)

        try:
            # Step 2: Execute the query
            rows = self.sql.read(sql_query)

            # Step 3: Use LLM to analyze results and provide intelligent response
            return self._analyze_results_and_respond(rows, action_type, original_query, intent)
        except Exception as e:
            return f"I encountered an error while retrieving the data: {str(e)}"

    def _generate_sql_for_action(self, action_type: str, intent: dict, original_query: str) -> str:
        """Generate SQL query using LLM understanding of the request."""
        # Database schema information for the LLM
        schema_info = """
        Available tables and columns:
        - customers: id, name, email, phone, created_at
        - orders: id, customer_id, total, status, created_at (JOIN with customers for customer_name)
        - order_items: id, order_id, product_id, quantity, price
        - products: id, sku, name, price, description
        - leads: id, customer_name, contact_email, message, score, status, created_at
        - tickets: id, customer_id, subject, message, status, priority, created_at
        """

        prompt = f"""
            You are a Sales & CRM agent generating SQL queries for a SQLite database.
            
            User query: "{original_query}"
            Action type: {action_type}
            Filters mentioned: {intent.get("filters", [])}
            Context: {intent.get("context", "")}
            
            {schema_info}
            
            Generate a SQL query to fulfill this request. Requirements:
            - Select appropriate columns for the action type
            - Use JOINs when needed (especially orders with customers)
            - Apply filters based on user intent (cancelled, recent, high_value, etc.)
            - Sort by most relevant field
            - Use ONLY SQLite-compatible syntax
            - For LIMIT values, use only integer numbers, not expressions
            - For "recent" or "latest" queries: Simply ORDER BY created_at DESC with LIMIT - do NOT use date filters
            - Only use date filters when user specifically mentions timeframes like "last week", "this month", etc.
            - Use appropriate LIMIT based on the request:
              * "all" or "every" = no LIMIT
              * "recent" or "latest" = LIMIT 10
              * "top" or "best" = LIMIT 10
              * "show me" = LIMIT 20
              * specific count mentioned = use that exact number
              * default = LIMIT 20
            
            Important: Respond with ONLY a valid SQL query, no backticks, no explanation, no formatting.
            Example formats:
            - Recent orders: SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id ORDER BY o.created_at DESC LIMIT 10
            - Show orders: SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id ORDER BY o.created_at DESC LIMIT 20
            """

        try:
            sql_query = call_gemini_prompt(prompt).strip()

            # Clean up the response
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:-3].strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query[3:-3].strip()
            return sql_query
        except Exception as e:
            # Fallback to basic query based on action type
            fallback_queries = {
                "retrieve_orders": "SELECT o.id, c.name AS customer_name, o.total, o.status, o.created_at FROM orders o JOIN customers c ON o.customer_id = c.id ORDER BY o.created_at DESC LIMIT 20",
                "retrieve_customers": "SELECT id, name, email, phone, created_at FROM customers ORDER BY created_at DESC LIMIT 20",
                "retrieve_products": "SELECT id, name, price, description FROM products ORDER BY price DESC LIMIT 20",
                "retrieve_leads": "SELECT id, customer_name, contact_email, score, status, created_at FROM leads ORDER BY created_at DESC LIMIT 20"
            }
            return fallback_queries.get(action_type, fallback_queries["retrieve_orders"])

    def _analyze_results_and_respond(self, rows: list, action_type: str, original_query: str, intent: dict) -> str:
        """Use LLM to analyze results and provide intelligent, contextual response with RAG context."""
        if not rows:
            return f"I couldn't find any data matching your criteria for: {original_query}"

        # Get contextual information from RAG as specified in requirements
        rag_context = ""
        if self.sales_rag:
            try:
                rag_results = self.sales_rag.search_procedures(original_query, k=2)
                if rag_results:
                    rag_context = f"\n\nRelevant procedures and context:\n"
                    for result in rag_results:
                        rag_context += f"- {result['excerpt']}\n"
            except Exception as e:
                print(f"Warning: RAG search failed: {e}")

        prompt = f"""
You are a Sales & CRM expert analyzing data for a business user.

User query: "{original_query}"
Action performed: {action_type}
Data found: {len(rows)} records
Sample data: {rows[:3] if len(rows) > 3 else rows}
{rag_context}

Provide a professional response that:
1. Directly answers their question
2. Summarizes key findings from the data
3. Highlights important business insights
4. Mentions any notable patterns or trends
5. Uses appropriate sales/CRM terminology
6. Incorporates relevant context from procedures when available

Be conversational, helpful, and business-focused.
"""

        return call_gemini_prompt(prompt)

    def _handle_rag_query(self, query: str) -> str:
        """Handle RAG queries for procedures, policies, definitions."""
        try:
            # Try domain-specific RAG first
            if self.sales_rag:
                rag_results = self.sales_rag.search(query, k=3)
                if rag_results:
                    return self._summarise_rag_results(rag_results, query, "sales procedures")

            # Fallback to general RAG
            rag_results = self.rag.search(query, k=3)
            if rag_results:
                return self._summarise_rag_results(rag_results, query, "documentation")

            return f"I couldn't find any relevant information about '{query}' in our sales documentation. You might want to contact your sales manager for specific procedures."

        except Exception as e:
            print(f"Warning: RAG search failed: {e}")
            return f"I'm having trouble accessing the documentation for '{query}'. Please try again or contact support."

    def _summarise_rag_results(self, results: list, query: str, source_type: str) -> str:
        """Summarize RAG results into a helpful response."""
        if not results:
            return f"I couldn't find information about '{query}' in our {source_type}."

        # Build context from results
        context = ""
        for i, result in enumerate(results[:3], 1):
            context += f"{i}. {result.get('excerpt', '')}\n"

        # Use LLM to provide a structured response
        prompt = f"""
Based on the following information from our {source_type}, answer the user's question: "{query}"

Relevant information:
{context}

Provide a clear, helpful response that addresses the user's question. Be professional and specific.
"""

        try:
            response = call_gemini_prompt(prompt)
            return response.strip()
        except Exception as e:
            # Fallback to simple concatenation
            return f"Based on our {source_type}:\n\n{context}"

    def _handle_creation_action(self, action_type: str, intent: dict, original_query: str) -> str:
        """Handle record creation actions using LLM to understand requirements."""
        if action_type == "new_lead":
            return self._create_intelligent_lead(original_query, intent)
        elif action_type == "new_order":
            return self._create_intelligent_order(original_query, intent)
        elif action_type == "support_ticket":
            return self._create_support_ticket(original_query, intent)
        else:
            return "I can help you create leads, orders, or support tickets. Could you be more specific?"

    def _create_intelligent_lead(self, original_query: str, intent: dict) -> str:
        """Create a lead using LLM to extract information with RAG context."""

        # Get relevant procedures for lead creation
        rag_context = ""
        if self.sales_rag:
            try:
                rag_results = self.sales_rag.search_procedures("lead qualification scoring", k=2)
                if rag_results:
                    rag_context = f"\n\nRelevant lead management procedures:\n"
                    for result in rag_results:
                        rag_context += f"- {result['excerpt']}\n"
            except Exception as e:
                print(f"Warning: RAG search failed: {e}")

        prompt = f"""
Extract lead information from this request:
"{original_query}"
{rag_context}

Carefully extract the specific details mentioned:
- Look for person names (first name + last name)
- Look for company names (often "at [Company]" or "[Company] Inc/LLC/Ltd")
- Look for email addresses (format: name@domain.com) - these are often after "email" or contain @
- Extract any message or description

Pay special attention to email addresses - they are critical for lead creation.

If specific details are provided, extract them exactly. If not provided, use reasonable defaults.
Consider the lead scoring criteria from procedures when available.

IMPORTANT: Respond with ONLY valid JSON, no explanations or markdown formatting.

JSON format:
{{
    "customer_name": "extracted full name (first last) or Company Name",
    "contact_email": "extracted email or reasonable default", 
    "message": "extracted message or description of the lead request",
    "score": 0.5
}}

Examples:
- "John Smith at TechCorp" -> customer_name: "John Smith", contact_email: "john.smith@techcorp.com"
- "Create lead for Jane Doe, email jane@example.com" -> customer_name: "Jane Doe", contact_email: "jane@example.com"
- "John Smith at TechCorp, email john@techcorp.com" -> customer_name: "John Smith", contact_email: "john@techcorp.com"
"""

        try:
            response = call_gemini_prompt(prompt)
            # Parse the lead info and create the lead
            import json

            # Clean up the response to extract JSON
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:-3].strip()
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:-3].strip()

            lead_info = json.loads(response_clean)

            customer_name = lead_info.get("customer_name", f"Lead {random.randint(1000, 9999)}")
            contact_email = lead_info.get("contact_email", f"lead{random.randint(1000, 9999)}@example.com")
            message = lead_info.get("message", "Lead created via chat interface")

            # Use ML tool for lead scoring
            scoring_result = LeadScoringTool.score_lead(customer_name, contact_email, message)
            ml_score = scoring_result.get("score", 0.5)

            # Create the lead in database
            sql = """
                INSERT INTO leads (customer_name, contact_email, message, score, status, created_at) 
                VALUES (?, ?, ?, ?, 'new', datetime('now'))
            """
            lead_id = self.sql.write(sql, (
                customer_name,
                contact_email,
                message,
                ml_score
            ))

            # Store lead scoring details in entity memory if we have a customer ID
            # For now, we'll just include the scoring info in the response
            confidence = scoring_result.get("confidence", "medium")
            reasoning = scoring_result.get("reasoning", "Standard lead scoring applied")

            return (f"Successfully created new lead (ID: {lead_id}) for {customer_name}. "
                    f"Lead score: {ml_score:.2f} (confidence: {confidence}). "
                    f"Scoring rationale: {reasoning}. The lead has been assigned for follow-up.")

        except Exception as e:
            # Fallback to simple lead creation
            return self._create_dummy_lead(original_query)

    def _create_intelligent_order(self, original_query: str, intent: dict) -> str:
        """Create an order using LLM to understand requirements."""
        # For now, use the existing dummy order creation
        # In a real system, this would parse customer/product info from the query
        return self._create_dummy_order(original_query)

    def _create_support_ticket(self, original_query: str, intent: dict) -> str:
        """Create a support ticket from the query."""
        prompt = f"""
Extract support ticket information from: "{original_query}"

Create JSON with:
{{
    "subject": "brief subject line",
    "message": "detailed message",
    "priority": "low|medium|high"
}}
"""

        try:
            response = call_gemini_prompt(prompt)
            import json
            ticket_info = json.loads(response.strip())

            # Create ticket (using customer_id = 1 as default)
            sql = """
                INSERT INTO tickets (customer_id, subject, message, status, priority, created_at)
                VALUES (?, ?, ?, 'open', ?, datetime('now'))
            """
            ticket_id = self.sql.write(sql, (
                1,  # Default customer
                ticket_info.get("subject", "Support Request"),
                ticket_info.get("message", original_query),
                ticket_info.get("priority", "medium")
            ))

            return f"Support ticket #{ticket_id} created successfully. Subject: {ticket_info.get('subject')}. Our team will respond soon."

        except Exception as e:
            return f"I've created a support ticket for your request: '{original_query}'. Our team will follow up with you soon."

    def _understand_intent(self, query: str) -> dict:
        """Use LLM to understand the user's intent and determine what they want."""
        prompt = f"""
        You are a sales domain expert. Analyze this user query and determine their intent.
        
        User query: "{query}"
        
        Available sales data includes: customers, leads, orders, order_items, products, tickets
        
        Determine:
        1. What action they want (query_data, create_record, or other)
        2. What entity they're interested in (customers, orders, products, leads, etc.)
        3. What specific information they need
        4. Any filters or conditions they mentioned
        
        Respond in JSON format:
        {{
            "action": "query_data|create_record|other",
            "entity": "customers|orders|products|leads|tickets",
            "information_needed": "description of what they want to know",
            "filters": ["list of any conditions or filters mentioned"],
            "limit": number or null
        }}
        """

        try:
            response = call_gemini_prompt(prompt)
            # Parse the JSON response
            import json
            return json.loads(response)
        except Exception as e:
            # Fallback to simple parsing
            return {
                "action": "query_data",
                "entity": "orders",
                "information_needed": "general information",
                "filters": [],
                "limit": 5
            }

    def _execute_data_query(self, intent: dict, original_query: str) -> str:
        """Execute a data query based on the understood intent."""
        entity = intent.get("entity", "orders")
        filters = intent.get("filters", [])
        limit = intent.get("limit", 5)

        # Generate SQL query based on intent
        sql_query = self._generate_sql_query(entity, filters, limit)

        try:
            # Execute the query
            rows = self.sql.read(sql_query)

            # Process and analyze the results intelligently
            return self._analyze_and_respond(rows, entity, original_query, intent)
        except Exception as e:
            return f"I encountered an error while retrieving the data: {str(e)}"

    def _generate_sql_query(self, entity: str, filters: list, limit: int) -> str:
        """Generate appropriate SQL query based on entity and filters."""
        base_queries = {
            "customers": """
                SELECT id, name, email, phone, created_at 
                FROM customers 
                ORDER BY created_at DESC
            """,
            "orders": """
                SELECT o.id, c.name AS customer_name, o.total, o.status, o.created_at
                FROM orders o 
                JOIN customers c ON o.customer_id = c.id
                ORDER BY o.created_at DESC
            """,
            "products": """
                SELECT id, name, price, description, sku
                FROM products
                ORDER BY price DESC
            """,
            "leads": """
                SELECT id, customer_name, contact_email, score, status, created_at
                FROM leads
                ORDER BY created_at DESC
            """,
            "tickets": """
                SELECT id, customer_id, subject, status, priority, created_at
                FROM tickets
                ORDER BY created_at DESC
            """
        }

        query = base_queries.get(entity, base_queries["orders"])

        # Apply filters if any
        if filters:
            # This is simplified - in a real system you'd parse filters more intelligently
            if "cancelled" in filters:
                query = query.replace("ORDER BY", "WHERE status = 'cancelled' ORDER BY")
            elif "recent" in filters:
                query = query.replace("ORDER BY", "WHERE created_at >= date('now', '-30 days') ORDER BY")

        # Add limit
        query += f" LIMIT {limit}"

        return query

    def _analyze_and_respond(self, rows: list, entity: str, original_query: str, intent: dict) -> str:
        """Intelligently analyze the data and provide a meaningful response."""
        if not rows:
            return f"I couldn't find any {entity} matching your criteria."

        # Create a summary of the data for the LLM to analyze
        data_summary = {
            "entity": entity,
            "count": len(rows),
            "data": rows[:10],  # Limit to first 10 rows for analysis
            "original_query": original_query,
            "intent": intent
        }

        prompt = f"""
        You are a sales analyst. The user asked: "{original_query}"
        
        Here is the data I found ({len(rows)} records):
        {data_summary}
        
        Provide a clear, helpful response that directly answers their question.
        Highlight key insights, trends, or important information.
        Be conversational and professional.
        """

        return call_gemini_prompt(prompt)

    def _execute_create_action(self, intent: dict, original_query: str) -> str:
        """Handle record creation actions."""
        # For now, use the existing dummy creation methods
        if "lead" in intent.get("entity", ""):
            return self._create_dummy_lead(original_query)
        elif "order" in intent.get("entity", ""):
            return self._create_dummy_order(original_query)
        else:
            return "I can help you create leads or orders. Could you be more specific about what you'd like to create?"

    # Helper methods
    def _summarise(self, raw_response: str, query: str) -> str:
        """Call the LLM to summarise a raw response string.

        Prepend a short instruction asking the model to present the data in
        a clear narrative.  Raises LLMError if the call fails.
        """
        prompt = (
            "You are an assistant summarising sales data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and friendly reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _list_customers(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, name, email, phone FROM customers ORDER BY created_at DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _list_leads(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, customer_name, contact_email, score, status FROM leads ORDER BY created_at DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _list_orders(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT orders.id, customers.name AS customer_name, orders.total, orders.status, orders.created_at "
            "FROM orders JOIN customers ON orders.customer_id = customers.id "
            "ORDER BY orders.created_at DESC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _list_products(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, sku, name, price FROM products ORDER BY id LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _create_dummy_lead(self, query: str) -> str:
        """Insert a dummy lead to demonstrate write operations.

        In a real implementation you would parse the user query for the
        customer name, email and message.  Here we generate placeholder
        values so that the end‑to‑end workflow can be exercised.  Returns
        a confirmation message summarised by the LLM.
        """
        name = f"Test Lead {random.randint(1000, 9999)}"
        email = f"test{random.randint(1000, 9999)}@example.com"
        message = "Generated by SalesAgent._create_dummy_lead"
        sql = (
            "INSERT INTO leads (customer_name, contact_email, message, score, status, created_at) "
            "VALUES (?, ?, ?, 0.0, 'new', datetime('now'))"
        )
        self.sql.write(sql, (name, email, message))
        raw_resp = {
            "lead_customer_name": name,
            "lead_contact_email": email,
            "lead_message": message,
        }
        return self._summarise(
            f"New lead created: {raw_resp}", query
        )

    def _create_dummy_order(self, query: str) -> str:
        """Insert a dummy order with a single line item.

        In a production system this would parse the query for customer and
        product identifiers.  Here we select a random customer and product
        from the database to create an example order.  Returns a summarised
        confirmation message.
        """
        # Pick a random customer
        customers = self.sql.read("SELECT id FROM customers")
        if not customers:
            raise RuntimeError("No customers available to create an order")
        customer_id = random.choice(customers)["id"]
        # Pick a random product
        products = self.sql.read("SELECT id, price FROM products")
        if not products:
            raise RuntimeError("No products available to create an order")
        product = random.choice(products)
        product_id = product["id"]
        price = product["price"]
        quantity = 1
        total = price * quantity
        # Create order header
        order_id = self.sql.write(
            "INSERT INTO orders (customer_id, total, status, created_at) VALUES (?, ?, 'pending', datetime('now'))",
            (customer_id, total),
        )
        # Create order item
        self.sql.write(
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
            (order_id, product_id, quantity, price),
        )
        raw_resp = {
            "order_id": order_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "quantity": quantity,
            "total": total,
        }
        return self._summarise(f"New order created: {raw_resp}", query)
