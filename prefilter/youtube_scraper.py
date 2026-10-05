import csv
import itertools
import logging
import os
import random
import time
from typing import Dict, List

import pandas as pd
import yt_dlp

from prefilter.config import (KEYWORDS_CSV, MAX_RESULTS_PER_QUERY, SCRAPE_DELAY_SECONDS,
                              SCRAPED_CSV, SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")


class VideoScraper:
    def __init__(self, csv_keyword_file=KEYWORDS_CSV, seed: int = SEED):
        self.logger = logging.getLogger(__name__)
        self.scraped_urls = set()
        self.queries = []
        self.fieldnames = ['query', 'platform', 'url', 'title', 'duration', 'view_count', 'description', 'uploader', 'upload_date']
        self.stats = {"queries_processed": 0, "youtube_videos": 0, "errors": 0}

        query_terms = pd.read_csv(csv_keyword_file)
        query_terms.fillna('', inplace=True)
        terms_emotion = set(query_terms.Emotion.unique())
        terms_subject = set(query_terms.Subject.unique())
        terms_setting = set(query_terms.Setting.unique())
        for s in [terms_emotion, terms_subject, terms_setting]:
            if '' in s:
                s.remove('')

        all_queries = list(itertools.product(sorted(terms_emotion), sorted(terms_setting), sorted(terms_subject)))
        random.Random(seed).shuffle(all_queries)

        for emo, setting, subj in all_queries:
            self.queries.append(f"{emo} {subj} {setting}".strip())

        self.logger.info(f"Initialized YouTube scraper with {len(self.queries)} queries")

    def make_csv_safe(self, text):
        if not text:
            return ""
        return "".join(char for char in str(text) if ord(char) <= 0xFFF)

    def scrape_youtube(self, delay: int = SCRAPE_DELAY_SECONDS, max_results: int = MAX_RESULTS_PER_QUERY,
                       output_file=SCRAPED_CSV) -> List[Dict]:
        all_videos = []

        done_queries: set[str] = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                done_queries = set(existing_df["query"].dropna().unique())
                self.scraped_urls = set(existing_df["url"].dropna().tolist())
                self.logger.info(
                    f"Resuming: {len(done_queries)} queries already done, "
                    f"{len(self.scraped_urls)} URLs already seen."
                )
            except Exception as e:
                self.logger.warning(f"Could not load existing CSV for resuming: {e}")
                self.scraped_urls = set()
        else:
            self.scraped_urls = set()

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
            "noplaylist": True,
            "ignoreerrors": True,
        }

        for query in self.queries:
            if query in done_queries:
                self.logger.info(f"[SKIP] Query already done: '{query}'")
                continue

            self.logger.info(f"Starting YouTube search: '{query}'")

            try:
                total_delay = delay + random.uniform(1, 2)
                time.sleep(total_delay)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)

                if not results:
                    self.logger.warning(f"No results for '{query}'")
                    continue

                entries = results.get("entries", []) or []

                videos: List[Dict] = []
                for entry in entries:
                    if not entry:
                        continue

                    video_id = entry.get("id")
                    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else entry.get("webpage_url")
                    if not url:
                        continue

                    if url in self.scraped_urls:
                        continue
                    self.scraped_urls.add(url)

                    videos.append({
                        "query": query,
                        "platform": "youtube",
                        "url": url,
                        "title": self.make_csv_safe(entry.get("title", "N/A")),
                        "duration": entry.get("duration") or 0,
                        "view_count": entry.get("view_count"),
                        "description": self.make_csv_safe(entry.get("description")),
                        "uploader": self.make_csv_safe(entry.get("uploader")),
                        "upload_date": entry.get("upload_date"),
                    })

                self.append_to_csv(output_file, videos)
                self.logger.info(f"Saved {len(videos)} videos for '{query}' to {output_file}")

                self.stats["youtube_videos"] += len(videos)
                self.stats["queries_processed"] += 1
                all_videos.extend(videos)

            except Exception as e:
                self.logger.error(f"YouTube search failed for '{query}': {e}", exc_info=True)
                self.stats["errors"] += 1
                continue

        self.logger.info(f"Done. Total: {len(all_videos)} videos, Errors: {self.stats['errors']}")
        return all_videos

    def append_to_csv(self, file_path: str, rows: List[Dict]):
        if not rows:
            return

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        file_exists = os.path.exists(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self.fieldnames})


if __name__ == "__main__":
    VideoScraper().scrape_youtube()
