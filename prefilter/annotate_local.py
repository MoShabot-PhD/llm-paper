import argparse

import ollama
import pandas as pd

from prefilter.annotate_api import write_input_csv
from prefilter.config import ANNOTATIONS_DIR, LOCAL_MODELS, LOCAL_OPTIONS, RESULTS_DIR
from prefilter.manifest import write_manifest
from prefilter.models.local import LocalModels

SMOKE_DIR = RESULTS_DIR / "smoke_test"


def check_model_digests():
    installed = {m.model: m.digest for m in ollama.list().models}
    for spec in LOCAL_MODELS.values():
        digest = installed.get(spec["tag"])
        if digest is None:
            raise RuntimeError(f"{spec['tag']} is not installed -- run: ollama pull {spec['tag']}")
        if digest != spec["digest"]:
            raise RuntimeError(f"{spec['tag']} has digest {digest}, expected {spec['digest']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Only 3 rows, written to results/smoke_test/")
    args = parser.parse_args()

    check_model_digests()

    input_csv = write_input_csv()
    out_dir = SMOKE_DIR if args.smoke_test else ANNOTATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        smoke_csv = out_dir / "to_annotate.csv"
        pd.read_csv(input_csv).head(3).to_csv(smoke_csv, index=False)
        input_csv = smoke_csv

    for name, spec in LOCAL_MODELS.items():
        print(f"=== {spec['tag']} ===")
        write_manifest(out_dir, spec["tag"], input_csv, {**LOCAL_OPTIONS, "digest": spec["digest"]})
        out = out_dir / f"annotations_metadata_{name}.csv"
        LocalModels(spec["tag"], str(input_csv)).run_annotations_kp(output_path=str(out))
        print(f"Output: {out}")


if __name__ == "__main__":
    main()
