#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
if __name__ != "__main__":
    __path__ = [str(Path(__file__).parent / "src" / "stockscreen")]
from stockscreen.server import main

if __name__ == "__main__":
    main()
