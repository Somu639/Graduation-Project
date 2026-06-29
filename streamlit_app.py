"""Streamlit Cloud entry point.

Streamlit executes THIS file on every rerun. It must call ``main()`` directly
(never via runpy) so all ``st.*`` widgets register with Streamlit's runner.
"""

from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

try:
    from processors.llm_client import bootstrap_env

    bootstrap_env()
except Exception:
    pass

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_FRONTEND = os.path.join(ROOT, "frontend")
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

os.environ["LOCAL_MODE"] = "1"
os.environ.setdefault("VECTOR_STORE", "memory")

_DASHBOARD_PATH = os.path.join(_FRONTEND, "streamlit_app.py")
_spec = importlib.util.spec_from_file_location("dashboard", _DASHBOARD_PATH)
_dashboard = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dashboard)
_dashboard.main()
