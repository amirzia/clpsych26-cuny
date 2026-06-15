"""Task 2 (Moments of Change) classifier: train, evaluate, and predict.

Trains a Switch ("S") or Escalation ("E") classifier on windowed per-post
features (see ``task2_features``) using nested cross-validation for model
selection and an unbiased performance estimate, then:

  * evaluates the production model against a tree of Task 1 LLM prediction
    folders, ranking each by F1 on the target label, and
  * writes a per-post submission prediction file.

Run for escalation:  ``python task2_classify.py --task E``
Run for switch:      ``python task2_classify.py --task S``
"""

import json
import argparse
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from clpsych.task2.features import (
    ALL_FEATURES,
    group_posts_by_timeline,
    select_features,
    flatten_windowed,
    flatten_labels,
    predict_sequence,
)
from clpsych.task2.predictions import inject_llm_presence

# ── Task configuration ──────────────────────────────────────────────────────

TASK_CONFIG = {
    "E": {
        "pos_label": "E",
        "neg_label": "0",
        "predict_label": "escalation",
        "target_names": ["Escalation", "No Escalation"],
        "window": 3,
    },
    "S": {
        "pos_label": "S",
        "neg_label": "0",
        "predict_label": "switch",
        "target_names": ["Switch", "No Switch"],
        "window": 2,
    },
}

RF_PARAM_GRID = {
    "clf__n_estimators": [100, 200, 500],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features": ["sqrt", "log2"],
}

SVC_PARAM_GRID = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__kernel": ["rbf", "linear"],
    "clf__gamma": ["scale", "auto"],
}

MODELS = [
    ("random_forest", RandomForestClassifier(class_weight="balanced", random_state=42), RF_PARAM_GRID),
    ("svm", SVC(class_weight="balanced", probability=True, random_state=42), SVC_PARAM_GRID),
]

INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# ── Data loading ────────────────────────────────────────────────────────────

def load_sequences(path: str | Path) -> list[list[dict]]:
    """Load a flat timeline JSON and group it into per-timeline sequences."""
    with open(path) as f:
        return group_posts_by_timeline(json.load(f))


def build_xy(
    sequences: list[list[dict]], feature_labels: dict[str, int], predict_label: str, window: int
) -> tuple[list[dict], list]:
    """Build the windowed feature matrix and label vector for training."""
    feature_seqs = [select_features(seq, feature_labels) for seq in sequences]
    label_seqs = [[post[predict_label] for post in seq] for seq in sequences]
    X = flatten_windowed(feature_seqs, window, feature_labels)
    y = flatten_labels(label_seqs)
    return X, y

# ── Model training ──────────────────────────────────────────────────────────

def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer()),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", clf),
    ])


def fit_best_model(clf, param_grid: dict, X, y, scorer) -> Pipeline:
    """Inner-CV grid search; returns the refit best estimator."""
    search = GridSearchCV(
        estimator=make_pipeline(clf),
        param_grid=param_grid,
        cv=INNER_CV,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)
    print("\n=== Best parameters for production ===")
    print(search.best_params_)
    print(f"Inner CV score: {search.best_score_:.3f}")
    return search.best_estimator_


def outer_cv_report(estimator: Pipeline, X, y, cfg: dict, model_name: str) -> None:
    """Aggregated outer-CV classification report (unbiased performance estimate)."""
    y_true, y_pred = [], []
    for train_idx, test_idx in OUTER_CV.split(X, y):
        estimator.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        y_true.extend(y[i] for i in test_idx)
        y_pred.extend(estimator.predict([X[i] for i in test_idx]))

    print(f"\n=== {model_name} - Aggregated Outer-CV Report ===")
    print(classification_report(
        y_true, y_pred,
        labels=[cfg["pos_label"], cfg["neg_label"]],
        target_names=cfg["target_names"],
    ))

# ── Evaluation against LLM prediction folders ───────────────────────────────

def evaluate_prediction_dir(
    group_dir: Path, pipe: Pipeline, cfg: dict, model_name: str,
    feature_labels: dict[str, int], base_timeline: list[dict], sort_list: list,
) -> None:
    """Score the model on every ``<shot>/results.csv`` under one model-config dir."""
    predict_label = cfg["predict_label"]
    labels = [cfg["pos_label"], cfg["neg_label"]]
    target_names = [predict_label, f"No {predict_label}"]

    for shot in sorted(p.name for p in group_dir.iterdir() if p.is_dir()):
        results_csv = group_dir / shot / "results.csv"
        if not results_csv.exists():
            continue

        adapted = inject_llm_presence(base_timeline, results_csv)
        y_true, y_pred = [], []
        for sequence in group_posts_by_timeline(adapted):
            predicted = predict_sequence(pipe, sequence, cfg["window"], feature_labels)
            for post, label in zip(sequence, predicted):
                if post["llm_predicted"]:
                    y_true.append(post[predict_label])
                    y_pred.append(label)

        report_kwargs = dict(labels=labels, target_names=target_names, zero_division=0.0)
        log_path = group_dir / shot / f"{predict_label}_{model_name}_log.txt"
        with open(log_path, "w") as f:
            f.write(classification_report(y_true, y_pred, **report_kwargs))

        report = classification_report(y_true, y_pred, output_dict=True, **report_kwargs)
        sort_list.append([f"{group_dir.name}/{shot}/{model_name}", report[predict_label]["f1-score"]])
        with open(group_dir / shot / f"{predict_label}_{model_name}_task_2_results.json", "w") as f:
            json.dump(report, f)


def evaluate_llm_folders(
    results_dir: Path, pipe: Pipeline, cfg: dict, model_name: str,
    feature_labels: dict[str, int], base_timeline: list[dict], sort_list: list,
) -> None:
    """Walk ``results_dir/<provider>/<model-config>/`` and evaluate each."""
    for provider in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for group_dir in sorted(p for p in provider.iterdir() if p.is_dir()):
            evaluate_prediction_dir(
                group_dir, pipe, cfg, model_name, feature_labels, base_timeline, sort_list
            )

# ── Submission ──────────────────────────────────────────────────────────────

def write_submission(
    pipe: Pipeline, cfg: dict, model_name: str, feature_labels: dict[str, int],
    submission_sequences: list[list[dict]], output_dir: Path,
) -> Path:
    """Predict the target label for every submission post and write the result file."""
    predict_label = cfg["predict_label"]
    results = []
    for sequence in submission_sequences:
        predicted = predict_sequence(pipe, sequence, cfg["window"], feature_labels)
        for post, label in zip(sequence, predicted):
            post[predict_label] = label
            results.append(post)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_{cfg['window']}_{predict_label}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    return out_path


def write_rankings(sort_list: list, predict_label: str, window: int, output_dir: Path) -> None:
    """Write LLM-folder F1 rankings, highest first."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"{predict_label}{window}.txt"
    print(f"\n==== {predict_label} w{window} Ranking ====")
    with open(ranking_path, "w") as f:
        for name, score in sorted(sort_list, key=lambda item: item[1], reverse=True):
            print(name, score)
            f.write(f"{name} {score}\n")

# ── Orchestration ───────────────────────────────────────────────────────────

def run(
    task: str,
    feature_labels: dict[str, int],
    data_source: Path,
    results_dir: Path,
    submission_csv: Path,
    timeline_all_path: Path,
    submission_input: Path | None = None,
    submission_output_dir: Path = Path("data/processed"),
    ranking_dir: Path = Path("processing"),
    models: list | None = None,
) -> None:
    """Run the full train / evaluate / submission pipeline for one task."""
    cfg = TASK_CONFIG[task]
    models = models or MODELS
    print(f"Features: {list(feature_labels.keys())}")

    scorer = make_scorer(f1_score, pos_label=cfg["pos_label"])

    X, y = build_xy(load_sequences(data_source), feature_labels, cfg["predict_label"], cfg["window"])

    with open(timeline_all_path) as f:
        base_timeline = json.load(f)

    # Submission timeline: a prebuilt adapted file if given, else the base
    # timeline overlaid with the submission CSV's LLM presence scores.
    if submission_input is not None:
        submission_sequences = load_sequences(submission_input)
    else:
        submission_sequences = group_posts_by_timeline(
            inject_llm_presence(base_timeline, submission_csv)
        )

    sort_list: list = []
    for model_name, clf, param_grid in models:
        best_model = fit_best_model(clf, param_grid, X, y, scorer)
        outer_cv_report(best_model, X, y, cfg, model_name)

        # Refit on all data for the production model used downstream.
        best_model = fit_best_model(clf, param_grid, X, y, scorer)
        evaluate_llm_folders(results_dir, best_model, cfg, model_name, feature_labels, base_timeline, sort_list)
        write_submission(best_model, cfg, model_name, feature_labels, submission_sequences, submission_output_dir)

    write_rankings(sort_list, cfg["predict_label"], cfg["window"], ranking_dir)

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and predict Task 2 moments of change.")
    parser.add_argument("--task", choices=["E", "S"], required=True, help="E for escalation, S for switch.")
    parser.add_argument(
        "--data-source", type=Path, required=True,
        help="Training timeline JSON (default: data/processed/timeline.json).",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root of LLM prediction folders to evaluate.",
    )
    parser.add_argument(
        "--submission-csv", type=Path, required=True,
        help="Task 1 results CSV used to build the submission timeline.",
    )
    parser.add_argument(
        "--timeline-all", type=Path, required=True,
        help="Base timeline JSON that LLM presence scores are overlaid onto.",
    )
    parser.add_argument(
        "--submission-input", type=Path, default=None,
        help="Optional prebuilt adapted timeline JSON to predict on (overrides --submission-csv overlay).",
    )
    parser.add_argument(
        "--features", nargs="+", choices=list(ALL_FEATURES), default=list(ALL_FEATURES),
        metavar="FEATURE", help=f"Feature subset to train on. Choose from: {list(ALL_FEATURES)}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        task=args.task,
        feature_labels={key: 0 for key in args.features},
        data_source=args.data_source,
        results_dir=args.results_dir,
        submission_csv=args.submission_csv,
        timeline_all_path=args.timeline_all,
        submission_input=args.submission_input,
    )
