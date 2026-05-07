from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data import ensure_demo_cache


if __name__ == "__main__":
    path = ensure_demo_cache()
    print(f"demo cache ready: {path}")
