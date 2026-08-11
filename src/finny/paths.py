"""Central path constants for Finny's data artifacts."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SUMMARIES_DIR = PROCESSED_DIR / "summaries"
DB_PATH = ROOT_DIR / "data" / "db" / "finny.sqlite"
CHROMA_DIR = ROOT_DIR / "data" / "index" / "chroma"
