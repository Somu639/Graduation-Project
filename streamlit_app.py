"""Root entry point for Streamlit Cloud.

Streamlit Community Cloud auto-detects a top-level ``streamlit_app.py``. This
launcher adds the repo root to the path and runs the real dashboard in
standalone (in-process) mode, so no separate FastAPI server is required.
"""

from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Default to the in-process backend on Streamlit Cloud (no API server there).
os.environ.setdefault("LOCAL_MODE", "1")

runpy.run_path(os.path.join(ROOT, "frontend", "streamlit_app.py"), run_name="__main__")
