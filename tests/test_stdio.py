import json
import subprocess
import sys
from pathlib import Path


def test_stdio_discovery_has_only_protocol_frames():
    process = subprocess.Popen(
        [sys.executable, "stockscreen.py"],
        cwd=Path(__file__).parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin and process.stdout
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    process.stdin.write(json.dumps(initialize) + "\n")
    process.stdin.flush()
    first = json.loads(process.stdout.readline())
    assert first["id"] == 1
    process.stdin.write(
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
    )
    process.stdin.write(
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n"
    )
    process.stdin.flush()
    second = json.loads(process.stdout.readline())
    assert second["id"] == 2 and {tool["name"] for tool in second["result"]["tools"]} == {
        "run_stock_screen",
        "get_stock_news",
        "manage_watchlist",
        "get_screening_result",
    }
    process.stdin.close()
    process.wait(timeout=15)
    assert process.returncode == 0
