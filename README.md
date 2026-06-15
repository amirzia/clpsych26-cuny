# CLPsych 2026 Shared Task

<div align="center">

<img src="asset/clpsych-logo.png" alt="CLPsych" height="80">
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="asset/cuny-graduate-center-logo.png" alt="CUNY Graduate Center" height="80">

<br><br>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.24164-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.24164)
[![Workshop](https://img.shields.io/badge/CLPsych-2026-6f42c1?style=for-the-badge)](https://clpsych.org)

**Official implementation of our submission to the [CLPsych 2026 Shared Task](https://clpsych.org) on modeling self-state dynamics ([paper](https://arxiv.org/abs/2605.24164))**

[🧭 Overview](#overview) • [⚙️ Requirements](#requirements) • [🚀 Usage](#usage) • [🔬 Data](#data) • [📊 Evaluation](#evaluation) • [📚 Citation](#citation)

</div>

## Overview

Code for our submission to the **CLPsych 2026 Shared Task** on modeling self-state
dynamics in Reddit posts using the **MIND / ABCD** psychological framework.

The repository covers three of the shared-task tracks:

- **Task 1 - Per-post self-state classification.** For each Reddit post, identify the
  adaptive and maladaptive self-states present. A self-state is a structured combination
  of six elements - Affect (A), Behavior toward others (B-O), Behavior toward the self
  (B-S), Cognition of others (C-O), Cognition of the self (C-S), and Desire (D). The
  model outputs the per-element subelement index and a presence rating for each state.

- **Task 2 - Moments of Change (MoC).** Over a chronological timeline, assign each post
  two independent binary labels: a *Switch* (`"S"`, a sudden well-being shift) and an
  *Escalation* (`"E"`, a gradual mood intensification). This is a classical ML pipeline
  (random forest / SVM over windowed self-state features), not LLM inference.

- **Task 3 - Sequence summary generation.** Write a structured narrative describing how
  self-state dynamics evolve across a chronological sequence of posts surrounding a
  clinical change event (a *Switch* or an *Escalation*).

LLM inference (Tasks 1 and 3) runs against an **OpenAI-compatible
[vLLM](https://github.com/vllm-project/vllm) endpoint** (default `http://127.0.0.1:8000/v1`).
There are no external API calls. You must serve the model locally and point the scripts at
it. Task 2 is CPU-only and consumes the presence scores produced by a Task 1 run.

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running scripts
- A running vLLM (or other OpenAI-compatible) server hosting the model you want to use

The code is organized as an installable package, `clpsych`, with one subpackage per
track (`clpsych.task1`, `clpsych.task2`, `clpsych.task3`) over a shared `clpsych.common`.
Dependencies (`openai`, `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `tqdm`)
are declared in `pyproject.toml`. `uv` resolves them and installs the package (editable)
into a project virtual environment - the first `uv run` below creates the environment, or
you can pre-build it:

```bash
uv sync
```

Each runnable script is a module, invoked with `uv run python -m clpsych.<task>.<module>`.

Embedding-based few-shot selection uses `BAAI/bge-large-en-v1.5`, which is downloaded
automatically from the Hugging Face Hub on first use.

### Serving a model

Start an OpenAI-compatible server before running any of the LLM scripts, for example:

```bash
vllm serve Qwen/Qwen3.5-27B --port 8000
```

Then pass `--model` and `--api_address` to match your server.

## Usage

### Task 1 - classification

Run inference (writes `results/<dir>/results.csv`):

```bash
uv run python -m clpsych.task1.run \
  --model Qwen/Qwen3.5-27B \
  --api_address http://127.0.0.1:8000/v1 \
  --mode few_shot --example_level post --sampling_type random --k 2 \
  --result_dir results/my-run
```

Prompting modes:

- `zero_shot` - task definition only
- `few_shot` + `--example_level subelement` - samples per-subelement evidence snippets
- `few_shot` + `--example_level post` - samples full example posts;
  `--sampling_type random` or `--sampling_type embedding` (cosine similarity over
  `BAAI/bge-large-en-v1.5` embeddings)

Convert the CSV to the competition submission JSON:

```bash
uv run python -m clpsych.task1.generate_submission \
  --input_path results/my-run/results.csv \
  --output_path results/my-run/submission.json
```

Ensemble several runs into a consensus submission. Pass two or more submission JSON
files (the ensemble members, each produced by `generate_submission` above) and the
combined prediction is built by **majority vote** on each subelement index and the
**median** of the presence ratings:

```bash
uv run python -m clpsych.task1.aggregate_results \
  --prediction_files results/gpt/submission.json results/gemma/submission.json results/qwen/submission.json \
  --output_path results/aggregate/submission.json
```

All input files must cover the same posts in the same order.

### Task 2 - moments of change

Task 2 is a three-stage CPU pipeline (no LLM serving required). It builds numeric
features from self-state presence scores, trains a Switch/Escalation classifier, and
merges the two label predictions into a submission file.

1. **Build the training feature table** from raw annotated timelines:

```bash
uv run python -m clpsych.task2.data --input data/raw-no-pred --output data/processed/timeline.json
```

2. **Train, evaluate, and predict** - run once per label (`E` = escalation, `S` = switch).
   This grid-searches a random forest and SVM with nested cross-validation, ranks every
   Task 1 LLM prediction folder by F1, and writes per-label submission predictions:

```bash
uv run python -m clpsych.task2.classify --task E
uv run python -m clpsych.task2.classify --task S
```

   Restrict the feature set with `--features` (choose from `adaptive`, `maladaptive`,
   `abs_adaptive`, `abs_maladaptive`, `count_adaptive`, `count_maladaptive`, `post_index`).

3. **Merge** the best escalation and switch result files into the final submission:

```bash
uv run python -m clpsych.task2.merge_submission \
  --esc    data/processed/random_forest_3_escalation_results.json \
  --switch data/processed/random_forest_2_switch_results.json \
  --output data/processed/task2_pred.json
```

`clpsych.task2.predictions` overlays a Task 1 `results.csv` onto a timeline; it is called
internally by `clpsych.task2.classify` and can also be run standalone.

### Task 3 - sequence summaries

Run inference (writes `results-task3/<dir>/predictions.json`):

```bash
uv run python -m clpsych.task3.run \
  --model openai/gpt-oss-120b \
  --api_address http://127.0.0.1:8000/v1 \
  --mode few_shot --k 3 \
  --result_dir results-task3/my-run
```

Ensemble judge - pick the best summary from multiple model outputs:

```bash
uv run python -m clpsych.task3.aggregate \
  --model openai/gpt-oss-120b \
  --api_address http://127.0.0.1:8000/v1 \
  --result_dir results-task3/aggregate-run \
  --example_summaries_files results-task3/run-a/predictions.json results-task3/run-b/predictions.json
```

Sequential, post-by-post pipeline:

```bash
uv run python -m clpsych.task3.summary_based \
  --model openai/gpt-oss-120b \
  --api-address http://127.0.0.1:8000/v1
```

LLM-based evaluation (consistency / contradiction vs. gold):

```bash
uv run python -m clpsych.task3.evaluate \
  --model openai/gpt-oss-120b \
  --prediction_file results-task3/my-run/predictions.json
```

### Common flags

- `--test` - switch any Task 3 script from the validation split to the held-out test set.
- `--debug` - print the full prompt and streamed response to stdout.

## Data

Data is **not** included in this repository (it is governed by the shared-task data use
agreement). To request access to the dataset, please contact the
[CLPsych 2026 workshop organizers](https://clpsych.org).

All data paths are CLI-overridable (`--train_dir`, `--test_dir`,
`--train_sequences_file`, `--test_sequences_file`). Training data is one JSON file per
timeline, named `<timeline_id>.json`. The train/val split is defined in
`clpsych/common/data_loader.py`.

## Evaluation

For evaluating the results on the validation set, please use the official scripts
[here](https://github.com/Maria-Liakata-NLP-Group/CLPsych-2026/tree/main).

## Citation

If you find this work useful, consider a ⭐️ and citing us with

```bibtex
@misc{bideh2026cunyclpsych2026pipeline,
      title={CUNY at CLPsych 2026: A Pipeline Approach to Classification and Summarization of Mental Health Changes}, 
      author={Amirmohammad Ziaei Bideh and Shameed Charlomar Job and Ava Yahyapour and Alla Rozovskaya},
      year={2026},
      eprint={2605.24164},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.24164}, 
}
```

## License

This project is released under the MIT License.
