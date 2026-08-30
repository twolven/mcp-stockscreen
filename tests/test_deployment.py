import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from stockscreen import server

ROOT = Path(__file__).parents[1]
LAUNCHER = "stockscreen.py"


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
            "host_origin_protection": True,
            "allowed_hosts": None,
            "allowed_origins": None,
            "show_banner": False,
        }
    ]


def test_host_origin_protection_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("MCP_HOST_ORIGIN_PROTECTION", raising=False)
    assert server.host_origin_protection() is True
    for value in ("true", "1", "on", "yes"):
        monkeypatch.setenv("MCP_HOST_ORIGIN_PROTECTION", value)
        assert server.host_origin_protection() is True
    for value in ("false", "0", "off", "no"):
        monkeypatch.setenv("MCP_HOST_ORIGIN_PROTECTION", value)
        assert server.host_origin_protection() is False
    monkeypatch.setenv("MCP_HOST_ORIGIN_PROTECTION", "auto")
    assert server.host_origin_protection() == "auto"
    monkeypatch.setenv("MCP_HOST_ORIGIN_PROTECTION", "bogus")
    with pytest.raises(ValueError, match="MCP_HOST_ORIGIN_PROTECTION"):
        server.host_origin_protection()


def test_allowlist_environment_is_parsed(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    assert server.csv_env("MCP_ALLOWED_HOSTS") is None
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", " , ")
    assert server.csv_env("MCP_ALLOWED_HOSTS") is None
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "a.example , b.example")
    assert server.csv_env("MCP_ALLOWED_HOSTS") == ["a.example", "b.example"]


def _post(url, host=None, origin=None):
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "guard-probe", "version": "1"},
            },
        }
    ).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json, text/event-stream")
    if host:
        request.add_header("Host", host)
    if origin:
        request.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_streamable_http_rejects_dns_rebinding():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "MCP_TRANSPORT": "streamable-http",
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, LAUNCHER],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("server did not become healthy")

        assert _post(f"{base}/mcp") == 200
        assert _post(f"{base}/mcp", origin=base) == 200
        assert _post(f"{base}/mcp", origin="http://evil.example") == 403
        assert _post(f"{base}/mcp", host="attacker.example") == 421
        assert (
            _post(f"{base}/mcp", host=f"evil.example:{port}", origin=f"http://evil.example:{port}")
            == 421
        )
    finally:
        process.terminate()
        process.wait(timeout=30)
