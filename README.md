# OWL Subsumption Axiom Refinement

A compact release repo for detecting suspicious OWL `subClassOf` axioms and repair. Current version is limited on the subsumption of atomic concept $A\sqsubseteq B$.

Default workflow:

1. Load an OWL ontology.
2. Fine-tune an Ontology Transformer (OnT) embedding model on that ontology.
3. Rank asserted direct superclass axioms and select the worst-ranked top K.
4. Validate those axioms with a local Qwen3-8B model.

## Quick Start

Run the example ontology with a Hugging Face OnT embedding model and skip LLM validation:

```bash
bash run.sh examples/custom_ontology.owl \
  --embedding-mode hf \
  --hf-model Hui97/OnT-MiniLM-L12-galen \
  --analyze-only \
  --top-n 5 \
  --candidate-k 3
```

This command creates the Python environment on first run, downloads or loads the Hugging Face OnT embedding model, ranks suspicious superclass axioms in `examples/custom_ontology.owl`, and writes outputs to `results/`. It does not require a local Qwen model or an API key because `--analyze-only` disables LLM validation.

`--analyze-only` means ranking-only mode. The pipeline still loads the ontology, computes OnT embeddings, ranks asserted `subClassOf` axioms, selects review candidates, and writes `rank_analysis.json`, `prompts.json`, `not_sent_to_llm.json`, `llm_review_results.json`, and `summary_report.md`. The difference is that it does not send prompts to a local or API LLM; reviewed items in `llm_review_results.json` are marked as skipped.

If no ontology path is provided, `run.sh` uses `examples/custom_ontology.owl`. The script uses conda if available, otherwise it creates `.venv`.

The local Qwen path is configured in `config.yaml` and should point to **your own downloaded model directory** when you run LLM validation:

```yaml
llm:
  backend: local
  local_model_path: models/Qwen3-8B
```

Use `SKIP_SETUP=1` when you have already activated a compatible Python environment:

```bash
SKIP_SETUP=1 bash run.sh examples/custom_ontology.owl --analyze-only
```

## Embedding Modes

Use your own fine-tuned OnT model:

```bash
bash run.sh ontology.owl \
  --embedding-mode model \
  --embedding-model models/ont_model/final
```

Fine-tune OnT from the input ontology, then run repair:

```bash
bash run.sh ontology.owl \
  --embedding-mode finetune \
  --finetune-epochs 3
```

Skip fine-tuning and use a Hugging Face OnT model:

```bash
bash run.sh ontology.owl \
  --embedding-mode hf \
  --hf-model Hui97/OnT-MiniLM-L12-galen
```

You can also fine-tune only:

```bash
python scripts/finetune_ont.py --owl ontology.owl --output runs/my_ont --epochs 3
```

## Ranking Parameters

```bash
bash run.sh ontology.owl --top-n 30 --candidate-k 10
```

Useful options:

- `--selection top-n` or `--selection threshold`
- `--top-n 30`
- `--threshold-k 20`
- `--candidate-k 10`
- `--analyze-only` to skip LLM validation

## LLM Backends

Default is local Qwen:

```bash
bash run.sh ontology.owl --llm-backend local --local-llm models/Qwen3-8B
```

Optional API backend, including OpenAI-compatible servers:

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
- `llm_review_results.json`: parsed LLM validation results.
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
  scripts/finetune_ont.py
  src/
    llm.py
    ont_model.py
    ontology_io.py
    prompts.py
    ranking.py
    run_pipeline.py
```

The fine-tuned model files are intentionally not committed. Provide a local path with `--embedding-model`, or publish the model on Hugging Face and pass the model ID.

## Notes

- `--analyze-only` skips LLM validation but still performs embedding-based ranking.
- `--embedding-mode finetune` writes model artifacts under `runs/` unless overridden.
- `--llm-backend local` first checks for an OpenAI-compatible vLLM server on `http://localhost:8000/v1`; otherwise it loads `local_model_path` with Transformers.
- CPU-only machines may need a custom PyTorch install. If `bitsandbytes` causes installation problems and you do not need quantized local LLM loading, remove it from the environment before installing.

## Tested Environment

This release was smoke-tested with a fresh conda environment created by `run.sh` on Linux x86_64. The validated command pattern was:

```bash
CONDA_ENV=axiom_repair_test bash run.sh examples/custom_ontology.owl \
  --embedding-mode hf \
  --hf-model Hui97/OnT-MiniLM-L12-galen \
  --analyze-only \
  --top-n 5 \
  --candidate-k 3
```

Tested software stack:

- Python 3.10 from the generated conda environment.
- `ontology-transformer 0.1.6`
- `rdflib 7.6.0`
- `torch 2.12.1+cu130`
- `sentence-transformers 5.6.0`
- `transformers 4.55.4`
- `openai 2.44.0`

The smoke test completed successfully on the example ontology in ranking-only mode with a downloadable Hugging Face OnT model. The test machine reported an older NVIDIA driver during CUDA initialization, so PyTorch fell back to a usable execution path for this small run; update the driver or install a matching PyTorch build for GPU-heavy fine-tuning or local LLM inference.

