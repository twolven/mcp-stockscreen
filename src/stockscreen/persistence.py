import json
import os
import re
import tempfile
from pathlib import Path

SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Store:
    def __init__(self, base: Path | None = None):
        self.base = (base or Path.home() / ".stockscreen").resolve()
        self.watchlists = self.base / "watchlists"
        self.results = self.base / "results"

    def _path(self, kind: str, name: str) -> Path:
        if not SAFE.fullmatch(name):
            raise ValueError("Name must contain only letters, digits, dot, underscore, or hyphen")
        root = self.watchlists if kind == "watchlist" else self.results
        return root / f"{name}.json"

    def save(self, kind, name, value):
        path = self._path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self, kind, name):
        path = self._path(kind, name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, kind, name):
        path = self._path(kind, name)
        if not path.exists():
            return False
        path.unlink()
        return True
