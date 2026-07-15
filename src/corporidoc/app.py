from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from corporidoc.data import PatientRepository
from corporidoc.ui import MainWindow


def default_data_dir() -> Path:
    configured = os.environ.get("CORPORIDOC_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".corporidoc"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CorporiDoC research application")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Local directory for the SQLite database and future artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("CorporiDoC")
    repository = PatientRepository(arguments.data_dir / "corporidoc.sqlite3")
    window = MainWindow(repository)
    window.show()
    return application.exec()
