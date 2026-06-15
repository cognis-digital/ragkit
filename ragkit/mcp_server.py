"""RAGKIT MCP server — exposes the RAG pipeline as an MCP tool for Cognis.Studio."""
from __future__ import annotations
from ragkit.core import answer, load_index

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
    def ragkit_query(index_path: str, query: str) -> str:
        """Query a ragkit index. index_path: path to index JSON; query: search question. Returns JSON."""
        import json as _json
        try:
            idx = load_index(index_path)
        except (FileNotFoundError, ValueError) as exc:
            return _json.dumps({"error": str(exc)})
        return _json.dumps(answer(idx, query))

    app.run()
    return 0
