import pandas as pd

from prefilter.config import EXPANSION_CSV, EXPANSION_QUERIES, TRIAGED_XLSX
from prefilter.youtube_scraper import VideoScraper


def seed_output_file():
    df = pd.read_excel(TRIAGED_XLSX)
    cols = ["query", "platform", "url", "title", "duration", "view_count",
            "description", "uploader", "upload_date"]
    EXPANSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(EXPANSION_CSV, index=False)
    print(f"Seed written: {len(df)} existing rows -> {EXPANSION_CSV}")
    return len(df)


def main():
    n_before = seed_output_file()

    scraper = VideoScraper()
    scraper.queries = scraper.queries[:EXPANSION_QUERIES]
    print(f"Running {len(scraper.queries)} queries")

    scraper.scrape_youtube(output_file=EXPANSION_CSV)

    n_after = len(pd.read_csv(EXPANSION_CSV, sep=None, engine="python", on_bad_lines="skip"))
    print(f"\nDone. Rows before: {n_before}, after: {n_after}, new: {n_after - n_before}")


if __name__ == "__main__":
    main()
