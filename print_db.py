"""Compatibility launcher for inspecting records: python print_db.py."""

import sys

from scripts import print_db as implementation

if __name__ == "__main__":
    implementation.display_all_records()
else:
    sys.modules[__name__] = implementation
