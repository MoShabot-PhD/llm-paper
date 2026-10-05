import pandas as pd

from prefilter.config import EXPANDED_XLSX, EXPANSION_CSV, TRIAGED_XLSX

ANNOTATION_COLS = [
    "RESULT METADATA", "RESULT VIDEO",
    "RESULT METADATA_JA", "RESULT VIDEO_JA",
    "RESULT METADATA_CAS", "RESULT VIDEO_CAS",
]


def main():
    orig = pd.read_excel(TRIAGED_XLSX)
    orig["url"] = orig["url"].astype(str).str.strip()

    exp = pd.read_csv(EXPANSION_CSV, sep=None, engine="python", on_bad_lines="skip")
    exp["url"] = exp["url"].astype(str).str.strip()

    new_rows = exp[~exp["url"].isin(set(orig["url"]))].copy()
    for col in ANNOTATION_COLS:
        new_rows[col] = pd.NA

    merged = pd.concat([orig, new_rows], ignore_index=True)
    assert merged["url"].is_unique, "Duplicate URLs after merge."

    merged.to_excel(EXPANDED_XLSX, index=False)
    print(f"{len(orig)} existing + {len(new_rows)} new = {len(merged)} rows -> {EXPANDED_XLSX}")


if __name__ == "__main__":
    main()
