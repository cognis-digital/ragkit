"""RAGKIT MCP server — exposes scan() as an MCP tool for Cognis.Studio."""
from __future__ import annotations
from ragkit.core import scan, to_json

def serve() -> int:
    """Start an MCP stdio server. Requires the optional 'mcp' extra:
        pip install "cognis-ragkit[mcp]"
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        print("Install the MCP extra: pip install 'cognis-ragkit[mcp]'")
        return 1
    app = FastMCP("ragkit")

    @app.tool()
    def ragkit_scan(target: str) -> str:
        """Batteries-included local RAG pipeline — ingest, index, serve. Returns JSON findings."""
        return to_json(scan(target))

    app.run()
    return 0
