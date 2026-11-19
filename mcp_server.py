
from mcp.server.fastmcp import FastMCP

from agent.operational_tools import get_transaction_details

# Initialize MCP server
mcp = FastMCP("Banking Transaction Server")


@mcp.tool()
def lookup_transaction(transaction_id: str) -> dict:
    """
    Retrieve transaction details using a transaction ID.

    Returns transaction information from the public sample dataset.
    Transaction statuses are synthetic demo data.
    """
    return get_transaction_details(transaction_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
