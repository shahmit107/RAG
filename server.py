"""Compatibility launcher: python server.py or uvicorn server:app."""

import sys

from backend import server as implementation

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(implementation.app, host="127.0.0.1", port=8000)
else:
    # Keep one module instance, including the shared pipeline lock and RAG state.
    sys.modules[__name__] = implementation
