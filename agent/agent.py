
from google.adk.agents import Agent
from predict_intent import predict_intent
from agent.operational_tools import get_transaction_details
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams
)
from mcp import StdioServerParameters
import sys
from pathlib import Path

def classify_banking_intent(question: str) -> dict:
    """
    Classify a banking employee's query using the
    fine-tuned DistilBERT BANKING77 model.
    """

    intent, confidence = predict_intent(question)

    return {
        "predicted_intent": str(intent),
        "confidence": round(float(confidence), 4)
    }


from document_engineering.retrieval.policy_router import (
    route_question,
    select_relevant_policies
)
from document_engineering.retrieval.generate_answer import generate_answer


def search_banking_policy(question: str) -> str:
    """
    Search approved banking policies and return an
    evidence-grounded answer to the employee's question.
    """

    # 1. Identify relevant policies
    routing_results = route_question(question)

    # 2. Select relevant policies
    selected_policies = select_relevant_policies(routing_results)

    # 3. Extract policy IDs
    selected_policy_ids = [
        policy["document_id"]
        for policy in selected_policies
    ]

    # 4. Generate answer using existing RAG pipeline
    answer = generate_answer(question, selected_policy_ids)

    return answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server.py"

transaction_mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER_PATH)]
        )
    )
)
root_agent = Agent(
    name="banking_operations_agent",
    model="gemini-3.8-flash",
    description="An AI assistant for banking operations.",
    instruction="""
        You are an Enterprise Banking Operations Copilot.

        MANDATORY WORKFLOW:

        For every banking-related query:

        1. First invoke classify_banking_intent.
        2. Read the predicted intent and confidence.
        3. If the user asks about banking policies or procedures,
        invoke search_banking_policy.
        4. Use the retrieved policy context to formulate your answer.
        5. If the user requests an operational action, use an
        appropriate authorized operational tool if available.
        6. Never invent banking procedures.
        7. Never request a customer's complete PIN or password.
        8.For transaction lookup requests, invoke get_transaction_details.
        Clearly disclose that transaction data comes from a public sample dataset
        and transaction statuses are synthetic demo data.
        Never present these as live bank records.

        Do not skip intent classification for banking queries.
        """,
        tools=[
            classify_banking_intent,
            search_banking_policy,
            transaction_mcp_toolset
        ]
)