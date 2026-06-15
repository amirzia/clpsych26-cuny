"""Feature engineering for Task 2 (Moments of Change).

Task 2 predicts two independent per-post binary labels over a chronological
timeline: ``Switch`` ("S"/"0") and ``Escalation`` ("E"/"0"). The classifier
operates on a small set of numeric per-post features (self-state presence
scores, subelement counts, consecutive deltas, post index) expanded into a
sliding context window so each prediction can see its neighbours.

This module is pure and importable: it contains no file I/O and no hardcoded
paths, so the same feature logic is shared by training (``task2_data``),
evaluation and submission (``task2_classify``).
"""

from typing import Any

# Per-post numeric features available to the classifier. The value (0) is the
# pad used for posts that fall outside the timeline when building context
# windows. Selecting a subset of these keys selects the active feature set.
ALL_FEATURES: dict[str, int] = {
    "abs_adaptive": 0,
    "abs_maladaptive": 0,
    "adaptive": 0,
    "maladaptive": 0,
    "count_adaptive": 0,
    "count_maladaptive": 0,
    "post_index": 0,
}

Post = dict[str, Any]


def group_posts_by_timeline(posts: list[Post]) -> list[list[Post]]:
    """Split a flat, timeline-ordered list of posts into per-timeline sequences.

    Posts must be contiguous by ``timeline_id`` (the order produced by
    ``task2_data``); a sequence boundary is started whenever the id changes.
    """
    sequences: list[list[Post]] = []
    current_timeline = object()  # sentinel that never equals a real id
    working: list[Post] = []
    for post in posts:
        if post["timeline_id"] != current_timeline:
            current_timeline = post["timeline_id"]
            sequences.append(working)
            working = []
        working.append(post)
    sequences.append(working)
    return sequences[1:]  # drop the empty list created before the first post


def add_timeline_deltas(timeline: list[Post]) -> list[Post]:
    """Annotate each post with the absolute change in presence vs. the previous post.

    ``abs_adaptive`` / ``abs_maladaptive`` are set to 0 whenever either post has
    a presence of 0 (i.e. the state is absent), otherwise the absolute
    difference. Mutates and returns ``timeline``.
    """
    if not timeline:
        return timeline
    previous = timeline[0]
    previous["abs_adaptive"] = 0
    previous["abs_maladaptive"] = 0
    for post in timeline[1:]:
        for state in ("adaptive", "maladaptive"):
            if previous[state] == 0 or post[state] == 0:
                post[f"abs_{state}"] = 0
            else:
                post[f"abs_{state}"] = abs(previous[state] - post[state])
        previous = post
    return timeline


def select_features(sequence: list[Post], feature_labels: dict[str, int]) -> list[Post]:
    """Keep only the active feature keys for each post in a sequence."""
    return [{key: post[key] for key in feature_labels} for post in sequence]


def make_windowed_features(
    sequence: list[Post], window: int, feature_labels: dict[str, int]
) -> list[dict[str, Any]]:
    """Expand each post into a flat feature dict spanning a +/- ``window`` context.

    Posts beyond the timeline edges are padded with ``feature_labels`` (zeros).
    Keys are namespaced by offset, e.g. ``offset_-1_adaptive``.
    """
    pad = [feature_labels] * window
    padded = pad + sequence + pad
    samples = []
    for i in range(window, len(padded) - window):
        flat: dict[str, Any] = {}
        for offset in range(-window, window + 1):
            neighbor = padded[i + offset]
            for key, value in neighbor.items():
                flat[f"offset_{offset}_{key}"] = value
        samples.append(flat)
    return samples


def flatten_windowed(
    sequences: list[list[Post]], window: int, feature_labels: dict[str, int]
) -> list[dict[str, Any]]:
    """Windowed features for every post across all sequences, as one flat list."""
    return [
        sample
        for sequence in sequences
        for sample in make_windowed_features(sequence, window, feature_labels)
    ]


def flatten_labels(label_sequences: list[list[Any]]) -> list[Any]:
    """Flatten per-sequence label lists into a single list, post order preserved."""
    return [label for sequence in label_sequences for label in sequence]


def predict_sequence(
    pipe, sequence: list[Post], window: int, feature_labels: dict[str, int]
) -> list[Any]:
    """Run a fitted sklearn pipeline over one timeline, returning per-post labels."""
    return pipe.predict(make_windowed_features(sequence, window, feature_labels))
