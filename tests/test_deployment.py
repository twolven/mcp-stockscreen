from pathlib import Path

from stockscreen import server

ROOT = Path(__file__).parents[1]


def test_container_is_non_root_streamable_http_deployment_with_persistence():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()
    assert "USER 10001:10001" in dockerfile
    assert "MCP_TRANSPORT=streamable-http" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '"127.0.0.1:${MCP_HOST_PORT:-8000}:8000"' in compose
    assert "stockscreen-data:/home/mcp/.stockscreen" in compose


def test_main_configures_streamable_http_from_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_PORT", "8123")
    monkeypatch.setenv("MCP_PATH", "/custom-mcp")
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: calls.append(kwargs))
    server.main()
    assert calls == [
        {
            "transport": "streamable-http",
            "host": "0.0.0.0",
            "port": 8123,
            "path": "/custom-mcp",
            "show_banner": False,
        }
    ]
