import subprocess
from pathlib import Path

wheel = next((Path(__file__).parents[1] / "dist").glob("*.whl"))
subprocess.run(
    ["uv", "run", "--isolated", "--with", str(wheel), "python", "-c", "import stockscreen.server"],
    check=True,
)
