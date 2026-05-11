import json
from pathlib import Path


def merge_shards(pattern: str = "sentiment_output_*.json", output_file: str = "sentiment_output_merged.json") -> int:
    """Merge shard JSON arrays into a single JSON array file."""
    parts = sorted(Path(".").glob(pattern))
    all_records = []

    for path in parts:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            all_records.extend(data)

    with Path(output_file).open("w", encoding="utf-8") as handle:
        json.dump(all_records, handle, ensure_ascii=False, indent=2)

    return len(all_records)


if __name__ == "__main__":
    count = merge_shards()
    print(f"Merged records: {count}")

