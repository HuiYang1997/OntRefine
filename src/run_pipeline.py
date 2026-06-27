from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import make_backend
from ont_model import finetune_ont
from ontology_io import load_owl
from prompts import build_prompt
from ranking import rank_axioms, select_for_review

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

ROOT = _SRC.parent


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def relpath(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return ROOT / path


def require_existing_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{description} not found: {path}')


def require_positive_int(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value}')
    return value


def override(value, current):
    return current if value is None else value


def parse_args():
    parser = argparse.ArgumentParser(description='OWL axiom repair with OnT ranking and Qwen/API validation')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--input', dest='input_owl', default=None, help='Input OWL file')
    parser.add_argument('--output', dest='output_dir', default=None, help='Output directory')

    parser.add_argument('--embedding-mode', choices=['model', 'finetune', 'hf'], default=None)
    parser.add_argument('--embedding-model', default=None, help='Local path or HF ID for an existing OnT model')
    parser.add_argument('--hf-model', default=None, help='HF model used when --embedding-mode hf')
    parser.add_argument('--finetune-output-dir', default=None)
    parser.add_argument('--finetune-epochs', type=int, default=None)
    parser.add_argument('--finetune-batch-size', type=int, default=None)
    parser.add_argument('--finetune-eval-batch-size', type=int, default=None)
    parser.add_argument('--finetune-balanced', action='store_true')
    parser.add_argument('--force-finetune', action='store_true')
    parser.add_argument('--skip-finetune', action='store_true', help='Use --hf-model instead of fine-tuning')

    parser.add_argument('--selection', choices=['top-n', 'threshold'], default=None)
    parser.add_argument('--top-n', type=int, default=None)
    parser.add_argument('--threshold-k', type=int, default=None)
    parser.add_argument('--candidate-k', type=int, default=None)

    parser.add_argument('--llm-backend', choices=['local', 'api', 'none'], default=None)
    parser.add_argument('--local-llm', default=None, help='Local Qwen model path')
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument('--api-base-url', default=None, help='OpenAI-compatible base URL')
    parser.add_argument('--api-key-env', default=None)
    parser.add_argument('--api-model', default=None)
    parser.add_argument('--analyze-only', action='store_true', help='Rank only; skip LLM validation')
    return parser.parse_args()


def group_to_dict(group) -> dict:
    return {
        'child_id': group.child_id,
        'parent_ids': group.parent_ids,
        'worst_rank': group.worst_rank,
        'pair_ranks': [
            {'parent_id': p.parent_id, 'rank': p.rank, 'score': round(p.score, 6)}
            for p in group.pair_ranks
        ],
        'alternatives': [
            {'class_id': cid, 'score': round(score, 6)}
            for cid, score in group.alternatives
        ],
    }


def write_summary(output_dir: Path, ontology_path: Path, model_ref: str, reviewed: list, skipped: list, llm_results: list) -> None:
    verdict_counts = {}
    for item in llm_results:
        verdict = item.get('llm_result', {}).get('overall_verdict', 'skipped')
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    lines = [
        '# Axiom Repair Summary',
        '',
        f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Ontology: `{ontology_path}`',
        f'Embedding model: `{model_ref}`',
        '',
        '## Counts',
        '',
        f'- Reviewed by LLM: {len(reviewed)}',
        f'- Not sent to LLM: {len(skipped)}',
    ]
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f'- {verdict}: {count}')
    lines += ['', '## Reviewed Axioms', '', '| Concept | Parents | Worst rank | Verdict | Summary |', '|---|---|---:|---|---|']
    for item in llm_results:
        group = item['rank']
        result = item.get('llm_result', {})
        verdict = result.get('overall_verdict', 'skipped')
        summary = result.get('summary', result.get('raw_response', ''))
        if not isinstance(summary, str):
            summary = ''
        lines.append(f"| `{group['child_id']}` | {', '.join(group['parent_ids'])} | {group['worst_rank']} | {verdict} | {summary[:140]} |")
    with open(output_dir / 'summary_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    emb_cfg = dict(cfg.get('embedding', {}))
    rank_cfg = dict(cfg.get('ranking', {}))
    llm_cfg = dict(cfg.get('llm', {}))

    input_owl = relpath(override(args.input_owl, cfg.get('input_owl', 'examples/custom_ontology.owl')))
    output_dir = relpath(override(args.output_dir, cfg.get('output_dir', 'results')))

    emb_cfg['mode'] = override(args.embedding_mode, emb_cfg.get('mode', 'finetune'))
    emb_cfg['model_path'] = override(args.embedding_model, emb_cfg.get('model_path', ''))
    emb_cfg['hf_model'] = override(args.hf_model, emb_cfg.get('hf_model', ''))
    emb_cfg['finetune_output_dir'] = override(args.finetune_output_dir, emb_cfg.get('finetune_output_dir', 'runs/ont_finetuned'))
    emb_cfg['finetune_epochs'] = override(args.finetune_epochs, emb_cfg.get('finetune_epochs', 3))
    emb_cfg['finetune_batch_size'] = override(args.finetune_batch_size, emb_cfg.get('finetune_batch_size', 64))
    emb_cfg['finetune_eval_batch_size'] = override(args.finetune_eval_batch_size, emb_cfg.get('finetune_eval_batch_size', 32))
    emb_cfg['finetune_balanced'] = bool(args.finetune_balanced or emb_cfg.get('finetune_balanced', False))
    emb_cfg['force_finetune'] = bool(args.force_finetune or emb_cfg.get('force_finetune', False))
    if args.skip_finetune and emb_cfg['mode'] == 'finetune':
        emb_cfg['mode'] = 'hf'

    rank_cfg['selection'] = override(args.selection, rank_cfg.get('selection', 'top-n'))
    rank_cfg['top_n'] = override(args.top_n, rank_cfg.get('top_n', 30))
    rank_cfg['threshold_k'] = override(args.threshold_k, rank_cfg.get('threshold_k', 20))
    rank_cfg['candidate_k'] = override(args.candidate_k, rank_cfg.get('candidate_k', 10))

    llm_cfg['backend'] = override(args.llm_backend, llm_cfg.get('backend', 'local'))
    llm_cfg['local_model_path'] = override(args.local_llm, llm_cfg.get('local_model_path', ''))
    llm_cfg['enable_thinking'] = bool(args.enable_thinking or llm_cfg.get('enable_thinking', False))
    llm_cfg['max_new_tokens'] = override(args.max_new_tokens, llm_cfg.get('max_new_tokens', 1500))
    llm_cfg['api_base_url'] = override(args.api_base_url, llm_cfg.get('api_base_url', ''))
    llm_cfg['api_key_env'] = override(args.api_key_env, llm_cfg.get('api_key_env', 'OPENAI_API_KEY'))
    llm_cfg['api_model'] = override(args.api_model, llm_cfg.get('api_model', 'gpt-4o-mini'))
    if args.analyze_only:
        llm_cfg['backend'] = 'none'

    require_existing_file(input_owl, 'Input OWL file')
    if emb_cfg['mode'] == 'hf' and not emb_cfg.get('hf_model'):
        raise RuntimeError('--embedding-mode hf requires --hf-model or embedding.hf_model in config.yaml')
    if llm_cfg.get('backend') == 'local':
        llm_cfg['local_model_path'] = str(relpath(llm_cfg.get('local_model_path', '')))
    rank_cfg['top_n'] = require_positive_int(rank_cfg['top_n'], 'top_n')
    rank_cfg['threshold_k'] = require_positive_int(rank_cfg['threshold_k'], 'threshold_k')
    rank_cfg['candidate_k'] = require_positive_int(rank_cfg['candidate_k'], 'candidate_k')

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Input OWL: %s', input_owl)
    logger.info('Output: %s', output_dir)
    concepts, groups = load_owl(input_owl)
    logger.info('Loaded %d classes and %d subsumption groups', len(concepts), len(groups))

    mode = emb_cfg['mode']
    if mode == 'model':
        model_ref = emb_cfg.get('model_path')
        if not model_ref:
            raise RuntimeError('--embedding-mode model requires --embedding-model')
    elif mode == 'hf':
        model_ref = emb_cfg.get('hf_model')
    elif mode == 'finetune':
        ft_dir = relpath(emb_cfg.get('finetune_output_dir'))
        model_ref = finetune_ont(
            owl_path=input_owl,
            output_dir=ft_dir,
            epochs=int(emb_cfg.get('finetune_epochs', 3)),
            batch_size=int(emb_cfg.get('finetune_batch_size', 64)),
            eval_batch_size=int(emb_cfg.get('finetune_eval_batch_size', 32)),
            balanced=bool(emb_cfg.get('finetune_balanced', False)),
            force=bool(emb_cfg.get('force_finetune', False)),
        )
    else:
        raise RuntimeError(f'Unknown embedding mode: {mode}')

    logger.info('Embedding model resolved to: %s', model_ref)
    ranks = rank_axioms(concepts, groups, model_ref=model_ref, candidate_k=rank_cfg['candidate_k'])
    reviewed, skipped = select_for_review(
        ranks,
        mode=rank_cfg['selection'],
        top_n=rank_cfg['top_n'],
        threshold_k=rank_cfg['threshold_k'],
    )

    rank_payload = {
        'ontology': str(input_owl),
        'embedding_mode': mode,
        'embedding_model': str(model_ref),
        'total_classes': len(concepts),
        'total_subsumption_groups': len(groups),
        'review_count': len(reviewed),
        'groups_sorted_by_worst_rank': [group_to_dict(g) for g in ranks],
    }
    with open(output_dir / 'rank_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(rank_payload, f, indent=2)

    prompts = [{'custom_id': g.child_id, 'messages': build_prompt(g, concepts, rank_cfg['candidate_k'])} for g in reviewed]
    with open(output_dir / 'prompts.json', 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2)

    llm_results = []
    if llm_cfg['backend'] == 'none':
        for group in reviewed:
            llm_results.append({'rank': group_to_dict(group), 'llm_result': {'skipped': True}})
    else:
        backend = make_backend(llm_cfg)
        for idx, group in enumerate(reviewed, 1):
            logger.info('LLM validation %d/%d: %s', idx, len(reviewed), group.child_id)
            messages = build_prompt(group, concepts, rank_cfg['candidate_k'])
            llm_results.append({'rank': group_to_dict(group), 'llm_result': backend.run_one(messages)})
            with open(output_dir / 'llm_review_results.partial.json', 'w', encoding='utf-8') as f:
                json.dump(llm_results, f, indent=2)

    with open(output_dir / 'llm_review_results.json', 'w', encoding='utf-8') as f:
        json.dump(llm_results, f, indent=2)
    with open(output_dir / 'not_sent_to_llm.json', 'w', encoding='utf-8') as f:
        json.dump([group_to_dict(g) for g in skipped], f, indent=2)
    write_summary(output_dir, input_owl, str(model_ref), reviewed, skipped, llm_results)
    logger.info('Done. Outputs written to %s', output_dir)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
