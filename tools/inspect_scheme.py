import sys
from pathlib import Path

# Standalone script (run as `python tools/inspect_scheme.py`).
_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))

from integrations import db

chunks = db.build_table_chunks()

for table, chunk in sorted(chunks.items()):
    print("\n" + "=" * 70)
    print(chunk)
