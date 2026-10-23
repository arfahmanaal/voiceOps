import os
import sys

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
DATASET_DIR = os.path.abspath(os.getenv("DATASET_DIR", "dataset"))
TODAY = "2026-10-20"

if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)