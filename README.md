# OWL Subsumption Axiom Refinement

A compact pipeline for finding suspicious asserted OWL `subClassOf` axioms and preparing repair suggestions. The current release focuses on atomic subsumption axioms of the form `A subClassOf B`.

## Quick Start

The main entry point is `run.sh`. Give it an ontology path; it will run the whole workflow:

```bash
bash run.sh examples/tiny_ontology.owl
```

By default, `run.sh` uses `embedding.mode: finetune` from `config.yaml`. This means it:

1. creates the Python environment if needed;
2. loads the input OWL ontology;
3. uses the `ontology-transformer` package to fine-tune an OnT embedding model on that ontology;
4. ranks asserted direct superclass axioms;
5. sends the selected candidates to the configured repair/validation backend.

The fine-tuned embedding model is written to `runs/ont_finetuned/` by default and reused on later runs. Use `--force-finetune` to train it again.

For a lightweight smoke test without LLM validation, keep the same automatic fine-tuning path and add `--analyze-only`:

```bash
bash run.sh examples/tiny_ontology.owl \
  --analyze-only \
  --finetune-epochs 1 \
  --top-n 3 \
  --candidate-k 2
```

`--analyze-only` only skips the LLM validation step. It still loads the ontology, fine-tunes/loads the OnT embedding model, ranks asserted `subClassOf` axioms, selects candidates, and writes all output files. Reviewed items in `llm_review_results.json` are marked as skipped.

## Configuration

Defaults live in `config.yaml`, and CLI flags override them. The most common settings are:

- `embedding.mode`: `finetune`, `model`, or `hf`.
- `embedding.finetune_epochs`: number of OnT fine-tuning epochs.
- `embedding.finetune_output_dir`: where the fine-tuned model is saved.
- `ranking.top_n`, `ranking.threshold_k`, `ranking.candidate_k`: controls candidate selection.
- `llm.backend`: `local`, `api`, or `none`.
- `llm.local_model_path`: relative path to your local Qwen model, for example `models/Qwen3-8B`.

## Embedding Modes

Default mode: fine-tune OnT on the input ontology and then run repair.

```bash
bash run.sh ontology.owl --embedding-mode finetune --finetune-epochs 3
```

Use your own already fine-tuned local OnT model:

```bash
bash run.sh ontology.owl \
  --embedding-mode model \
  --embedding-model models/ont_model/final
```

Use a Hugging Face OnT model directly, without ontology-specific fine-tuning:

```bash
bash run.sh ontology.owl \
  --embedding-mode hf \
  --hf-model <hf-ont-model-id>
```

Fine-tune only, without running the repair pipeline:

```bash
python scripts/finetune_ont.py --owl ontology.owl --output runs/my_ont --epochs 3
```

## Ranking And Repair Options

```bash
bash run.sh ontology.owl --top-n 30 --candidate-k 10
```

Useful options:

- `--selection top-n` or `--selection threshold`
- `--top-n 30`
- `--threshold-k 20`
- `--candidate-k 10`
- `--analyze-only` to skip LLM validation while still running embedding and ranking

Local Qwen validation:

```bash
bash run.sh ontology.owl --llm-backend local --local-llm models/Qwen3-8B
```

OpenAI or OpenAI-compatible API validation:

```bash
OPENAI_API_KEY=... bash run.sh ontology.owl \
  --llm-backend api \
  --api-model gpt-4o-mini
```

For another OpenAI-compatible endpoint:

```bash
MY_API_KEY=... bash run.sh ontology.owl \
  --llm-backend api \
  --api-key-env MY_API_KEY \
  --api-base-url http://localhost:8000/v1 \
  --api-model Qwen3-8B
```

## Outputs

Outputs are written to `results/` by default:

- `rank_analysis.json`: ranked suspicious axioms and candidate parents.
- `prompts.json`: exact LLM prompts.
- `llm_review_results.json`: parsed LLM validation results, or skipped markers in `--analyze-only` mode.
- `not_sent_to_llm.json`: axioms not sent for validation.
- `summary_report.md`: short report.

## Repo Structure

```text
axiom_repair_github_release/
  config.yaml
  environment.yml
  requirements.txt
  run.sh
  examples/custom_ontology.owl
  examples/tiny_ontology.owl
  scripts/finetune_ont.py
  src/
    llm.py
    ont_model.py
    ontology_io.py
    prompts.py
    ranking.py
    run_pipeline.py
```

The fine-tuned model files and generated outputs are intentionally not committed.

## Tested Environment

Smoke-tested on Linux x86_64 with a fresh conda environment created by `run.sh`:

```bash
CONDA_ENV=axiom_repair_test bash run.sh examples/tiny_ontology.owl \
  --analyze-only \
  --finetune-epochs 1 \
  --top-n 3 \
  --candidate-k 2
```

Tested software stack:

- Python 3.10 from the generated conda environment.
- `ontology-transformer 0.1.6`
- `rdflib 7.6.0`
- `torch 2.12.1+cu130`
- `sentence-transformers 5.6.0`
- `transformers 4.55.4`
- `openai 2.44.0`

The smoke test completed successfully on `examples/tiny_ontology.owl` in `--analyze-only` mode. For GPU-heavy fine-tuning or local LLM inference, install a PyTorch/CUDA build that matches your driver.
