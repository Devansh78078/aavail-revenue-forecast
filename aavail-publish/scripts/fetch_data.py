"""Fetch the public IBM capstone dataset at a pinned upstream revision."""
import argparse
from io import BytesIO
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

REVISION = "fd6182c4458bccb56024047a1a85f506a90caf98"
SOURCE = f"https://codeload.github.com/aavail/ai-workflow-capstone/zip/{REVISION}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    with urlopen(SOURCE, timeout=60) as response, ZipFile(BytesIO(response.read())) as archive:
        counts = {"cs-train": 0, "cs-production": 0}
        for entry in archive.infolist():
            parts = Path(entry.filename).parts
            if len(parts) != 3 or parts[1] not in counts or not parts[2].endswith(".json"):
                continue
            target = args.output / parts[1] / parts[2]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            counts[parts[1]] += 1
    if counts != {"cs-train": 21, "cs-production": 5}:
        raise RuntimeError(f"Unexpected dataset contents: {counts}")
    print(f"Fetched {sum(counts.values())} monthly files into {args.output}")


if __name__ == "__main__":
    main()
