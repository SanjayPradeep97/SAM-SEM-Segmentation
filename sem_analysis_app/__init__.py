"""
Gradio web application for SEM/TEM particle analysis.

Layout:
    state.py       the shared AppState instance
    callbacks/     event handlers, one module per tab
    ui.py          interface definition and event wiring
    visualization.py  overlay and figure rendering

Run it with ``python -m sem_analysis_app`` from the repository root.
"""

from .ui import create_interface

__all__ = ["create_interface"]
