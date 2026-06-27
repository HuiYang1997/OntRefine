from __future__ import annotations

from dataclasses import dataclass, field
import logging

from ont_model import OnTModel
from ontology_io import ConceptInfo, SubsumptionGroup

logger = logging.getLogger(__name__)


@dataclass
class PairRank:
    child_id: str
    parent_id: str
    rank: int
    score: float


@dataclass
class GroupRank:
    child_id: str
    parent_ids: list[str]
    pair_ranks: list[PairRank] = field(default_factory=list)
    worst_rank: int = 0
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def _top_k(scores: dict[str, float], exclude: set[str], k: int) -> list[tuple[str, float]]:
    items = [(cid, score) for cid, score in scores.items() if cid not in exclude]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def rank_axioms(
    concepts: dict[str, ConceptInfo],
    groups: list[SubsumptionGroup],
    model_ref: str,
    candidate_k: int = 10,
) -> list[GroupRank]:
    model = OnTModel(model_ref)
    candidate_ids = sorted(concepts)
    texts = [concepts[cid].text_for_embedding() for cid in candidate_ids]
    logger.info('Encoding %d ontology classes', len(candidate_ids))
    embeddings_raw = model.encode_tensor(texts)
    embeddings = {cid: embeddings_raw[i] for i, cid in enumerate(candidate_ids)}

    results: list[GroupRank] = []
    for idx, group in enumerate(groups):
        if idx % 50 == 0:
            logger.info('Ranking progress: %d/%d', idx, len(groups))
        child_emb = embeddings.get(group.child_id)
        if child_emb is None:
            continue
        scores = {}
        for candidate_id in candidate_ids:
            if candidate_id == group.child_id:
                continue
            scores[candidate_id] = model.score_subsumption(child_emb, embeddings[candidate_id])
        sorted_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        rank_map = {cid: i + 1 for i, cid in enumerate(sorted_ids)}
        pair_ranks = [
            PairRank(group.child_id, parent_id, rank_map.get(parent_id, len(sorted_ids)), scores.get(parent_id, 0.0))
            for parent_id in group.parent_ids
        ]
        result = GroupRank(
            child_id=group.child_id,
            parent_ids=list(group.parent_ids),
            pair_ranks=pair_ranks,
            worst_rank=max((p.rank for p in pair_ranks), default=0),
            alternatives=_top_k(scores, set(group.parent_ids) | {group.child_id}, candidate_k),
        )
        results.append(result)

    results.sort(key=lambda r: r.worst_rank, reverse=True)
    return results


def select_for_review(results: list[GroupRank], mode: str, top_n: int, threshold_k: int) -> tuple[list[GroupRank], list[GroupRank]]:
    if mode == 'top-n':
        return results[:top_n], results[top_n:]
    if mode == 'threshold':
        to_review = [r for r in results if r.worst_rank > threshold_k]
        skipped = [r for r in results if r.worst_rank <= threshold_k]
        return to_review, skipped
    raise ValueError(f'Unknown selection mode: {mode}')
