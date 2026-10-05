import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from prefilter.prompts import PROMPT

PACKAGES = ["pandas", "openpyxl", "yt-dlp", "requests", "python-dotenv", "ollama"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(out_dir: Path, model: str, input_csv: Path, params: dict) -> Path:
    path = out_dir / f"manifest_{re.sub(r'[^a-zA-Z0-9_-]', '_', model)}.json"
    if path.exists():
        return path

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None

    manifest = {
        "model": model,
        "params": params,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "input_csv": str(Path(input_csv).name),
        "input_sha256": file_sha256(input_csv),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
