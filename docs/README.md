# bbleqa

Benchmarking utilities for legal QA evaluation.

## Build Context

`generation.cli.build_context` converts retrieved article ids into per-question context JSON and retrieval metrics.

### CLI

Preferred (installed entrypoint):

```bash
poetry run bbleqa-build-context \
  --retrieved-file data/retrieved/voyage3_test_nl.json \
  --lang nl \
  --split test \
  --output-dir data/context \
  --overwrite
```

Module fallback:

```bash
PYTHONPATH=src python3 -m generation.cli.build_context \
  --retrieved-file data/retrieved/voyage3_test_nl.json \
  --lang nl \
  --split test \
  --output-dir data/context \
  --overwrite
```

Required flag:

- `--retrieved-file`: JSON keyed by question id, where each value is a retrieved article-id list
  (for example `{ "696": ["3604", "15393", ...] }`)

Useful optional flags:

- `--output-dir` (default: `data/context`)
- `--dataset-id` (default: `clips/bLLeQA`)
- `--split` (default: `test`)
- `--lang` (`nl` or `fr`, default: `nl`) used to compute retrieval metrics and gold-injection IDs
- `--overwrite`

Outputs:

- `data/context/<retrieved-file-stem>_metrics.json`
- `data/context/nl/<retrieved-file-stem>_only_gold.json`
- `data/context/nl/<retrieved-file-stem>_top_100.json`
- `data/context/nl/<retrieved-file-stem>_top_100_plus_gold.json`
- `data/context/fr/<retrieved-file-stem>_only_gold.json`
- `data/context/fr/<retrieved-file-stem>_top_100.json`
- `data/context/fr/<retrieved-file-stem>_top_100_plus_gold.json`

Each context output is keyed by question id and contains:

- `[{"id": "<article_id>", "text": "<article_text>"}, ...]`
- `text` is built as `"<reference>\n<article>"`

## Generate

`generation.cli.generate` runs answer generation for the bLLeQa split using a context JSON keyed by question id.

### Requirements

- Python `>=3.11,<3.12`
- Project dependencies installed (`poetry install`)
- OpenRouter API key in `OPENROUTER_API_KEY` (or in model config JSON)

### CLI

Preferred (installed entrypoint):

```bash
poetry run bbleqa-generate \
  --context-file data/context/nl/voyage3_test_nl_top_100.json \
  --model ministral-8b \
  --lang nl
```

Module fallback:

```bash
PYTHONPATH=src python3 -m generation.cli.generate \
  --context-file data/context/nl/voyage3_test_nl_top_100.json \
  --model ministral-8b \
  --lang nl
```

Required flags:

- `--context-file`: JSON file containing context keyed by bLLeQa question id
- `--model`: model config filename stem from `src/generation/model_configs` (for example `ministral-8b`)

Useful optional flags:

- `--config-dir` (default: `src/generation/model_configs`)
- `--outputs-dir` (default: `outputs`)
- `--dataset-id` (default: `clips/bLLeQa_aligned`)
- `--split` (default: `test`)
- `--lang` (`nl` or `fr`, default: `nl`)
- `--max-concurrency` (default: `5`)

### Output

Output is written to:

- `outputs/<lang>/<context-file-stem>/<model>.json`

If that file already exists, completed ids in `answers` are skipped on rerun.

## Evaluate Faithfulness

`generation.cli.evaluate_faitfulness` computes RAGAS faithfulness metrics from generation outputs.

### CLI

Run from project root:

```bash
PYTHONPATH=src python3 -m generation.cli.evaluate_faitfulness \
  --generation-dir outputs \
  --context-dir data/context \
  --output-dir outputs/results \
  --model ministral-8b \
  --setup voyage3_test_nl_top_100 \
  --lang nl
```

Required flags:

- `--generation-dir`: directory containing generation output files
- `--model`: one or more model names
- `--setup`: one or more setup names (context filename stem)
- `--lang`: `nl` or `fr`

Useful optional flags:

- `--context-dir` (default: `data/context`)
- `--output-dir` (default: `outputs/results`)
- `--dataset-id` (default: `clips/bLLeQa_aligned`)
- `--split` (default: `test`)
- `--judge-model` (default: `gemini-3-flash`)
- `--ragas-batch-size` (default: `5`)
- `--no-cache`

Notes:

- Faithfulness contexts are built from cited source ids in model outputs.
- Per cited article, context text is built as `"<reference>\n<article>"`.
- Maximum cited contexts passed to the evaluator per question: `100`.

Output:

- `outputs/results/<lang>/<setup>/<model>/ragas.json`

## Judge Selector

`generation.judge_selector` compares multiple judge models on the same calibration JSON using DeepEval correctness, then writes a JSON ranking.

### What It Produces

- `selected_judge_name`
- `selected_judge_config`
- `ranking` (all candidates with metrics)

Per-candidate metrics include:

- `coverage`
- `failure_rate`
- `num_scored`
- `num_failed`
- `mean_score`
- `score_std`
- `failed_eval_ids`
- `failure_reasons`
- agreement metrics computed from CSV `grade`:
- `pearson_correlation`
- `spearman_correlation`
- `accuracy`
- `f1_score`
- `mean_true_binary`
- `mean_pred_binary`

### How It Works

1. Load judge configs from JSON files in `src/generation/model_configs`.
2. Load rows from JSON with fixed keys: `ids,questions,gold_answers,llm_answers,grade`.
3. Run `deepeval_correctness` for each judge on the same rows.
   If cached per-model correctness exists in `outputs/judge_selection/<input-file-stem>/<modelname>.json`,
   only not-yet-evaluated examples are sent for new evaluation.
4. Save all intermediate per-model correctness outputs to an intermediate JSON file.
5. Compute ranking metrics only if every model finished with zero failed evaluations.
6. Save output JSON.

### Requirements

- Python environment with project dependencies installed.
- OpenRouter API key available via `OPENROUTER_API_KEY` or inside each model config file.

### CLI

Run from project root:

```bash
PYTHONPATH=src python3 -m generation.judge_selector \
  --csv-path data/annotations/data-annotations-fr.json \
  --candidates gemma-3-27b-it \
  --output-file outputs/judge_selection/judge_selection.json
```

### Output File

Default output path:

- `outputs/judge_selection/judge_selection.json`

Intermediate output path:

- `outputs/judge_selection/judge_selection_intermediate.json`

Per-model correctness outputs:

- `outputs/judge_selection/<input-file-stem>/<modelname>.json`

### Model Config Files

Each file in `src/generation/model_configs/*.json` defines one judge candidate.

Example:

```json
{
  "name": "google/gemma-3-27b-it",
  "providers": ["novita/bf16"]
}
```
