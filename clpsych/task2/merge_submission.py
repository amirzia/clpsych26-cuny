"""Merge per-label Task 2 predictions into a single submission file.

``task2_classify`` writes separate escalation and switch result files (one per
model/window). This step pairs them post-by-post into the ``task2_pred.json``
submission format defined by the shared task: one entry per post with both a
``Switch`` ("S"/"0") and an ``Escalation`` ("E"/"0") label.
"""

import json
import argparse
from pathlib import Path


def merge_submissions(esc_path: Path, switch_path: Path, output_path: Path) -> None:
    with open(esc_path) as f:
        esc_data = json.load(f)
    with open(switch_path) as f:
        switch_data = json.load(f)

    output = []
    for esc_post, switch_post in zip(esc_data, switch_data):
        if esc_post["post_id"] != switch_post["post_id"]:
            raise ValueError(
                f"Post ID mismatch: {esc_post['post_id']} vs {switch_post['post_id']}"
            )
        output.append({
            "timeline_id": esc_post["timeline_id"],
            "post_id": esc_post["post_id"],
            "Switch": switch_post["switch"],
            "Escalation": esc_post["escalation"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"Wrote {len(output)} posts to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge escalation and switch predictions into a single submission file."
    )
    parser.add_argument(
        "--esc", type=Path, default=Path("data/processed/random_forest_3_escalation_results.json"),
        help="Escalation results JSON from task2_classify.",
    )
    parser.add_argument(
        "--switch", type=Path, default=Path("data/processed/random_forest_2_switch_results.json"),
        help="Switch results JSON from task2_classify.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/task2_pred.json"),
        help="Output path for the merged submission JSON (default: data/processed/task2_pred.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_submissions(esc_path=args.esc, switch_path=args.switch, output_path=args.output)
