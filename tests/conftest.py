from pathlib import Path
import sys

# The repository's top-level Streamlit app imports local modules from the
# project root. Keep that import path available in CI, where PYTHONPATH=src
# is also set for the research package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
