"""Project paths shared by the service and command-line scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DOCUMENTS_DIR = PROJECT_ROOT / "Pdfs"
DATABASE_DIR = PROJECT_ROOT / "my_local_db"
UPLOADS_DIR = PROJECT_ROOT / "storage" / "documents"
