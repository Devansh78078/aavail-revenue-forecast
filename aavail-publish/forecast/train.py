"""Train a model bundle and export expanding-window validation results."""
import argparse
from pathlib import Path

from .ingest import ingest
from .model import fit, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    bundle, validation = fit(ingest(*args.sources))
    save_model(bundle, args.model)
    if args.validation:
        args.validation.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(args.validation, index=False)
    print(f"Trained {bundle['version']} across {len(bundle['models'])} scopes")


if __name__ == "__main__":
    main()
