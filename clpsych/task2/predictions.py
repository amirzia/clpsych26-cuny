"""Inject Task 1 LLM presence predictions into a Task 2 timeline.

Task 2 can be evaluated either on gold self-state presence scores or on the
presence scores predicted by a Task 1 LLM run. This module reads a Task 1
``results.csv`` and overlays its ``llm_adaptive_presence`` /
``llm_maladaptive_presence`` columns onto a base timeline, keeping only the
posts the LLM actually scored.

The core helper ``inject_llm_presence`` works in memory; a thin CLI is kept for
running the merge standalone.
"""

import csv
import json
import argparse
from pathlib import Path


def read_llm_presence(csv_path: str | Path) -> dict[str, dict[str, int]]:
    """Read a Task 1 results CSV into ``{post_id: {adaptive, maladaptive}}``."""
    presence: dict[str, dict[str, int]] = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            presence[row["post_id"]] = {
                "adaptive": int(row["llm_adaptive_presence"]),
                "maladaptive": int(row["llm_maladaptive_presence"]),
            }
    return presence


def inject_llm_presence(timeline_posts: list[dict], csv_path: str | Path) -> list[dict]:
    """Overlay LLM presence onto ``timeline_posts``, keeping only scored posts.

    Each returned post is flagged ``llm_predicted=True`` and has its
    ``adaptive`` / ``maladaptive`` presence replaced by the LLM prediction.
    Input order is preserved, so the result stays grouped by timeline.
    """
    presence = read_llm_presence(csv_path)
    adapted = []
    for post in timeline_posts:
        if post["post_id"] in presence:
            post = {**post, "llm_predicted": True, **presence[post["post_id"]]}
            adapted.append(post)
    return adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Task 1 LLM presence predictions onto a Task 2 timeline."
    )
    parser.add_argument("csv_path", type=Path, help="Task 1 results CSV with LLM presence columns.")
    parser.add_argument(
        "--timeline-path", type=Path, default=Path("data/processed/timeline_all.json"),
        help="Source timeline JSON (default: data/processed/timeline_all.json).",
    )
    parser.add_argument(
        "--timeline-output", type=Path, default=Path("data/processed/timeline_adapted.json"),
        help="Output path for the merged timeline JSON (default: data/processed/timeline_adapted.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.timeline_path) as f:
        timeline_posts = json.load(f)
    adapted = inject_llm_presence(timeline_posts, args.csv_path)
    args.timeline_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.timeline_output, "w") as f:
        json.dump(adapted, f)
    print(f"Wrote {len(adapted)} LLM-scored posts to {args.timeline_output}")
