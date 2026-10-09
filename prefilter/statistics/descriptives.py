import re

import pandas as pd

from common import HUMAN_ANNOTATIONS_CSV, OUT_DIR, VIDEO_RATERS, load_evaluation_set, normalize

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"#\w+")


def metadata_properties(df: pd.DataFrame) -> pd.DataFrame:
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    features = {
        "Title length (characters)": title.str.len(),
        "Title length (words)": title.str.split().str.len(),
        "Description length (characters)": desc.str.len(),
        "Description length (words)": desc.str.split().str.len(),
        "URLs in description": desc.str.count(URL_RE),
        "Hashtags in description": desc.str.count(HASHTAG_RE),
        "Duration (minutes)": df["duration"] / 60,
        "View count (thousands)": df["view_count"] / 1000,
        "Upload year": df["upload_date"].astype(str).str[:4].astype(int),
    }
    rows = []
    for name, s in features.items():
        s = s.astype(float)
        rows.append({"feature": name, "mean": s.mean(), "sd": s.std(ddof=1), "median": s.median(),
                     "q1": s.quantile(0.25), "q3": s.quantile(0.75)})
    return pd.DataFrame(rows)


def metadata_vs_video(df: pd.DataFrame) -> pd.DataFrame:
    for r in VIDEO_RATERS:
        df[f"video_{r}"] = df[f"video_{r}"].map(normalize)
    df = df.dropna(subset=[f"video_{r}" for r in VIDEO_RATERS])

    def majority(prefix: str) -> pd.Series:
        votes = sum((df[f"{prefix}_{r}"] == "KEEP").astype(int) for r in VIDEO_RATERS)
        return pd.Series(votes > len(VIDEO_RATERS) / 2, index=df.index).map({True: "KEEP", False: "DISCARD"})

    table = pd.crosstab(majority("meta").rename("metadata"), majority("video").rename("video"))
    agreement = (majority("meta") == majority("video")).mean()
    print(f"\nMetadata vs video majority labels (n = {len(df)}), agreement = {agreement:.1%}\n{table}")
    return table


def main():
    all_videos = pd.read_csv(HUMAN_ANNOTATIONS_CSV)
    properties = metadata_properties(all_videos)
    print(f"Metadata properties (n = {len(all_videos)})\n{properties.to_string(index=False, float_format='{:,.1f}'.format)}")

    evaluation, _ = load_evaluation_set()
    discrepancy = metadata_vs_video(evaluation.copy())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    properties.to_csv(OUT_DIR / "metadata_properties.csv", index=False)
    discrepancy.to_csv(OUT_DIR / "metadata_vs_video.csv")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
