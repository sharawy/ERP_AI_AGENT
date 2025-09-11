"""
FastAPI application for the agent‑driven ERP system.

This module wires together the router and domain agents and exposes a
/chat endpoint for conversational interactions.  It also includes a few
sample endpoints for listing domain data directly.  The agents
communicate with the database via SQL helpers and summarise their results
using the LLM.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from agents import (
    SalesAgent,
    FinanceAgent,
    InventoryAgent,
    AnalyticsAgent,
)
from agents.react_router_agent import LangchainRouter
from models.common import ChatRequest, ChatResponse, ListOrdersResponse, CreateOrderRequest, CreateOrderResponse
from tools.sql_tool import SQLTool
from tools.memory_manager import conversation_manager
from tools.audit_logger import GlobalState


app = FastAPI(title="Agent‑Driven ERP API")
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
# Instantiate agents once at startup

langchain_router = LangchainRouter()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Entry point for conversational interactions with the ERP system.

    The router selects the appropriate domain agent based on the user
    message and returns the agent's response.  Conversation history is
    recorded in the database.
    """
    try:
        # Get or create conversation ID
        conversation_id = req.conversation_id
        if not conversation_id:
            conversation_id = conversation_manager.start_conversation()

        # Add user message to conversation buffer
        conversation_manager.add_user_message(conversation_id, req.message)

        # Route the query using LangChain ReAct agent
        result = langchain_router.run(req.message)

        # Add agent response to conversation buffer
        conversation_manager.add_agent_response(conversation_id, result, "system")

        # Update global state
        GlobalState.set_last_module("router")

    except Exception as e:
        print(e)
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(conversation_id=conversation_id, response=result)


@app.get("/orders", response_model=ListOrdersResponse)
def list_orders(limit: int = 10) -> ListOrdersResponse:
    """List the most recent sales orders."""
    sql = SQLTool("api")
    rows = sql.read(
        "SELECT orders.id, customers.name AS customer_name, orders.total, orders.status, orders.created_at "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "ORDER BY orders.created_at DESC LIMIT ?",
        (limit,),
    )
    return ListOrdersResponse(orders=rows)


@app.post("/orders", response_model=CreateOrderResponse)
def create_order(req: CreateOrderRequest) -> CreateOrderResponse:
    """Create a new order with given items.

    This endpoint demonstrates how to create orders directly via the API
    without going through the chat interface.  It assumes that each item
    has keys `product_id` and `quantity`.  The total is computed based on
    the product price.  Returns the new order ID.
    """
    sql = SQLTool("api")
    # Validate customer
    customer = sql.read("SELECT id FROM customers WHERE id = ?", (req.customer_id,))
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    # Compute total
    total = 0.0
    for item in req.items:
        product = sql.read("SELECT price FROM products WHERE id = ?", (item["product_id"],))
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {item['product_id']} not found")
        price = product[0]["price"]
        total += price * item["quantity"]
    # Insert order
    order_id = sql.write(
        "INSERT INTO orders (customer_id, total, status, created_at) VALUES (?, ?, 'pending', datetime('now'))",
        (req.customer_id, total),
    )
    # Insert line items
    for item in req.items:
        product = sql.read("SELECT price FROM products WHERE id = ?", (item["product_id"],))
        price = product[0]["price"]
        sql.write(
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
            (order_id, item["product_id"], item["quantity"], price),
        )
    return CreateOrderResponse(order_id=order_id, message="Order created successfully")