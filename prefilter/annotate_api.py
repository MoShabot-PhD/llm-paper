import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from prefilter.config import (ANNOTATIONS_DIR, API_CHUNK_SIZE, API_MAX_TOKENS, API_MODELS, API_TEMPERATURE,
                              TO_ANNOTATE_CSV, TRIAGED_XLSX)
from prefilter.manifest import write_manifest
from prefilter.models.openrouter import OpenRouterModel


def write_input_csv():
    df = pd.read_excel(TRIAGED_XLSX)
    TO_ANNOTATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TO_ANNOTATE_CSV, index=False)
    print(f"{len(df)} rows -> {TO_ANNOTATE_CSV}")
    return TO_ANNOTATE_CSV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Only 3 rows, to check model ids and prompt")
    args = parser.parse_args()

    load_dotenv()
    input_csv = write_input_csv()
    model = OpenRouterModel(
        api_key=os.environ["OPENROUTER_API_KEY"],
        yt_path=str(input_csv),
        max_rows=3 if args.smoke_test else None,
    )
    params = {"temperature": API_TEMPERATURE, "max_tokens": API_MAX_TOKENS, "chunk_size": API_CHUNK_SIZE}
    for name in API_MODELS:
        print(f"\n=== {name} ===")
        write_manifest(ANNOTATIONS_DIR, name, input_csv, params)
        out = model.run_batch_classifier(platform="youtube", model=name, batch_dir=str(ANNOTATIONS_DIR),
                                         chunk_size=API_CHUNK_SIZE)
        print(f"Output: {out}")


if __name__ == "__main__":
    main()
