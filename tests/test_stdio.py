import json
import subprocess
import sys
from pathlib import Path


def test_stdio_discovery_has_only_protocol_frames():
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    result = subprocess.run(
        [sys.executable, "stockscreen.py"],
        cwd=Path(__file__).parents[1],
        input="".join(json.dumps(x) + "\n" for x in messages),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert {frame.get("id") for frame in frames} == {1, 2}
    assert all(line.startswith("{") for line in result.stdout.splitlines())
