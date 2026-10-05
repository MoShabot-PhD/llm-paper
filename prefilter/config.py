from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

KEYWORDS_CSV = ROOT / "keywords.csv"
SCRAPED_CSV = DATA_DIR / "yt_scraped_exp.csv"
TRIAGED_XLSX = DATA_DIR / "triaged_videos.xlsx"
EXPANSION_CSV = DATA_DIR / "youtube_videos_expansion.csv"
EXPANDED_XLSX = DATA_DIR / "triaged_videos_expanded.xlsx"
TO_ANNOTATE_CSV = RESULTS_DIR / "to_annotate.csv"
ANNOTATIONS_DIR = RESULTS_DIR / "annotations"

SEED = 42

MAX_RESULTS_PER_QUERY = 10
SCRAPE_DELAY_SECONDS = 7
EXPANSION_QUERIES = 130

API_MODELS = ["anthropic/claude-sonnet-4.6:batch", "openai/gpt-5.2:batch"]
API_TEMPERATURE = 1
API_MAX_TOKENS = 512
API_CHUNK_SIZE = 300

LOCAL_MODELS = {
    "qwen3_6": {
        "tag": "qwen3.6:latest",
        "digest": "07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
    },
    "gemma4_31b": {
        "tag": "gemma4:31b",
        "digest": "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7",
    },
}
LOCAL_OPTIONS = {"temperature": 0.3, "top_p": 0.9, "seed": SEED}
