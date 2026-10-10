"""Compatibility launcher: streamlit run streamlit_app.py."""

import runpy

runpy.run_module("frontend.streamlit_app", run_name="__main__")
