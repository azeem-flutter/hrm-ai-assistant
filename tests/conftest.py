"""Shared pytest setup.

config/config.py exits the process if the Oracle env vars are missing, so we
provide harmless dummy values BEFORE any project module is imported. The unit
tests never open a database connection or call an LLM.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("DB_DSN")
os.environ.setdefault("RAG_PROVIDER", "keyword")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
