"""
Inventory & Supply Chain agent.

This agent manages stock levels, suppliers and purchase orders.  It
interprets user queries with simple keyword heuristics and calls the
database to retrieve or modify data.  After performing an operation it
invokes the LLM to summarise the result for the user.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from agents.BaseAgent import BaseAgent
from llm import call_gemini_prompt, LLMError
from tools.sql_tool import SQLTool
from tools.vector_rag_tool import VectorRAGTool, DocRAGTool


class InventoryAgent(BaseAgent):
    def __init__(self) -> None:
        self.sql = SQLTool("inventory")
        self.rag = VectorRAGTool("main_documents")  # General RAG
        self.domain = 'inventory'
        # Domain-specific RAG tools as per requirements
        try:
            self.doc_rag = DocRAGTool()
        except Exception as e:
            print(f"Warning: Could not initialize DocRAGTool: {e}")
            self.doc_rag = None

    def get_default_intent_prompt(self, query: str) -> str:
        """Return a default system prompt for intent analysis."""
        intent_prompt = ChatPromptTemplate.from_template(
            """You are an assistant for the {domain} domain.
            Analyze the user's query and classify the type of action needed.
            query: {query}
            Respond with ONLY a JSON object:
            {{
                "action_type": "retrieve_data|create_record|update_record|delete_record|restock|other",
                "filters": ["list of filters or conditions"],
                "context": "brief description of what the user wants"
            }}
            Additional Notes:
            - Use 'restock' as the action_type if the user mentions restocking, reordering, or replenishing stock.
            - Ensure to identify specific SKUs or products if mentioned in the query.
            - If no SKU is provided, clarify by asking which SKU the user wants to restock."""
        )
        messages = intent_prompt.format_messages(domain=self.domain, query=query)
        return messages[0].content

    def handle_query(self, query: str, conversation_id: Optional[int] = None) -> str:
        try:
            # Step 1: Understand what the user wants
            intent = self._analyze_intent(query)
            logging.debug(intent)
            # Step 2: Process based on the intent action_type
            action_type = intent.get("action_type", "other")
            filters = intent.get("filters", [])
            context = intent.get("context", "")

            # Step 3: Route to the appropriate handler based on action_type
            if action_type == "retrieve_data":
                # Handle data retrieval based on context
                if any(term in context.lower() for term in ["stock", "inventory", "levels"]):
                    return self._list_stock(query)
                elif any(term in context.lower() for term in ["product", "price", "pricing", "expensive", "cheapest"]):
                    return self._list_products_by_price(query)
                elif "supplier" in context.lower():
                    return self._list_suppliers(query)
            elif action_type == "create_record" and any(term in context.lower() for term in ["purchase order", "po"]):
                return self._create_dummy_po(query)
            elif action_type == "restock":
                import re
                skus = re.findall(r"SKU-\d{4}", query.upper())
                if not skus:
                    return self._summarise(f"Which SKU do you want to restock?", query)
                sku = skus[0]
                logging.info("Restock request for SKU: %s", sku)
                prod = self.sql.read("SELECT id FROM products WHERE sku=:s", {"s": sku})
                if not prod:  return self._summarise(f"no product found {sku}", query)
                pid = prod[0]["id"]
                stock = self.sql.read("SELECT qty_on_hand, reorder_point FROM stock WHERE product_id=:p", {"p": pid})[0]
                if stock["qty_on_hand"] < stock["reorder_point"]:
                    # Create PO to bring it to 2x reorder
                    target = stock["reorder_point"] * 2
                    qty = max(0, target - stock["qty_on_hand"])
                    self.sql.write("INSERT INTO purchase_orders (supplier_id, status) VALUES (1, 'draft')", {})
                    po_id = self.sql.read("SELECT last_insert_rowid() AS id")[0]["id"]
                    self.sql.write(
                        "INSERT INTO po_items (po_id, product_id, quantity, unit_cost) VALUES (:po,:p,:q, 0.6)",
                        {"po": po_id, "p": pid, "q": qty})
                    return self._summarise(
                        f"PO #{po_id} created to restock {sku} by {qty} units. data:po_id:{po_id}", query)
                else:
                    return self._summarise(f"{sku} is above reorder point. No action needed.", query)

            # Fallback RAG search
            rag_results = self.rag.search(query)
            if rag_results:
                return self._summarise(
                    f"RAG search results for '{query}': {rag_results}", query
                )
            return "I'm sorry, I couldn't understand your inventory request."
        except Exception as e:
            logging.error(e)
            raise

    def _summarise(self, raw_response: str, query: str) -> str:
        prompt = (
            "You are an inventory assistant summarising data for a user.  "
            "Given the following raw information and the user's query, "
            "compose a concise and helpful reply.\n\n"
            f"User query: {query}\n"
            f"Raw information: {raw_response}"
        )
        return call_gemini_prompt(prompt)

    def _list_stock(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT products.name AS product_name, stock.qty_on_hand, stock.reorder_point "
            "FROM stock JOIN products ON stock.product_id = products.id "
            "ORDER BY stock.qty_on_hand ASC LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _list_products_by_price(self, query: str) -> str:
        q = query.lower()
        if "expensive" in q or "highest" in q:
            # Show most expensive products
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price DESC LIMIT 5"
            )
        elif "cheapest" in q or "lowest" in q:
            # Show cheapest products
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price ASC LIMIT 5"
            )
        else:
            # Show products with pricing info
            rows = self.sql.read(
                "SELECT products.name, products.price, products.description, stock.qty_on_hand "
                "FROM products LEFT JOIN stock ON products.id = stock.product_id "
                "ORDER BY products.price DESC LIMIT 5"
            )
        return self._summarise(str(rows), query)

    def _list_suppliers(self, query: str) -> str:
        rows = self.sql.read(
            "SELECT id, name, email, phone FROM suppliers ORDER BY name LIMIT 5"
        )
        return self._summarise(str(rows), query)

    def _create_dummy_po(self, query: str) -> str:
        """Create a dummy purchase order for demonstration.

        Select a random supplier and product and create a purchase order with
        a single line item.  Returns a summarised confirmation message.
        """
        suppliers = self.sql.read("SELECT id FROM suppliers")
        if not suppliers:
            raise RuntimeError("No suppliers available to create a purchase order")
        supplier_id = random.choice(suppliers)["id"]
        # Pick a random product
        products = self.sql.read("SELECT id, price FROM products")
        if not products:
            raise RuntimeError("No products available to create a purchase order")
        product = random.choice(products)
        product_id = product["id"]
        unit_cost = product["price"]  # using selling price as stand‑in for cost
        quantity = random.randint(1, 10)
        # Create PO header
        po_id = self.sql.write(
            "INSERT INTO purchase_orders (supplier_id, status, created_at) VALUES (?, 'draft', datetime('now'))",
            (supplier_id,),
        )
        # Create PO line
        self.sql.write(
            "INSERT INTO po_items (po_id, product_id, quantity, unit_cost) VALUES (?, ?, ?, ?)",
            (po_id, product_id, quantity, unit_cost),
        )
        raw_resp = {
            "po_id": po_id,
            "supplier_id": supplier_id,
            "product_id": product_id,
            "quantity": quantity,
            "unit_cost": unit_cost,
        }
        return self._summarise(f"New purchase order created: {raw_resp}", query)
