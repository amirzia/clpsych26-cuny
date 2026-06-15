"""Build the Task 2 training feature table from raw annotated timelines.

Reads one raw timeline JSON per file and flattens every post into a record
carrying the numeric features used by the classifier (see ``task2_features``)
plus the gold ``switch`` / ``escalation`` labels. The output is a single flat,
timeline-ordered JSON list consumed by ``task2_classify``.

This stage parses the raw annotation JSON directly rather than going through
``data_loader`` on purpose: Task 2 needs presence encoded as 0 when a state is
absent (``data_loader`` defaults a present-but-unscored state to 1) and needs
the raw "S"/"E"/"0" label strings rather than booleans.
"""

import json
import argparse
from pathlib import Path

from clpsych.task2.features import add_timeline_deltas


def post_to_record(post: dict, timeline_id: str) -> dict:
    """Flatten one raw annotated post into a feature/label record."""
    record = {
        "timeline_id": timeline_id,
        "post_id": post["post_id"],
        "post_index": post["post_index"],
        "wellbeing": post.get("Well-being") or 0,
        "switch": post["Switch"],
        "escalation": post["Escalation"],
        "adaptive": 0,
        "maladaptive": 0,
        "count_adaptive": 0,
        "count_maladaptive": 0,
    }

    evidence = post.get("evidence")
    if evidence:
        # Number of annotated subelements per state (excluding the Presence key).
        record["count_adaptive"] = max(0, len(evidence["adaptive-state"]) - 1)
        record["count_maladaptive"] = max(0, len(evidence["maladaptive-state"]) - 1)
        if "Presence" in evidence["adaptive-state"]:
            record["adaptive"] = evidence["adaptive-state"]["Presence"]
            record["maladaptive"] = evidence["maladaptive-state"]["Presence"]

    return record


def build_training_table(raw_dir: Path) -> list[dict]:
    """Flatten every timeline in ``raw_dir`` into a single ordered record list."""
    table: list[dict] = []
    timeline_count = 0
    post_count = 0

    for file_path in sorted(raw_dir.glob("*.json")):
        with open(file_path) as f:
            timeline = json.load(f)

        timeline_id = file_path.stem
        timeline_count += 1
        records = [post_to_record(post, timeline_id) for post in timeline["posts"]]
        post_count += len(records)
        table.extend(add_timeline_deltas(records))

    print(f"Built {post_count} posts across {timeline_count} timelines")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task 2 timeline training data from raw annotation files."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw-no-pred"),
        help="Folder of raw timeline JSON files (default: data/raw-no-pred).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/timeline.json"),
        help="Output path for the processed timeline JSON (default: data/processed/timeline.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = build_training_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=4)
        f.write("\n")