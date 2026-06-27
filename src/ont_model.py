from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def choose_model_dir(path: str | Path) -> str:
    path = Path(path).expanduser()
    if (path / 'final').exists():
        return str(path / 'final')
    return str(path)


def finetune_ont(
    owl_path: str | Path,
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 64,
    eval_batch_size: int = 32,
    balanced: bool = False,
    force: bool = False,
) -> str:
    output_dir = Path(output_dir)
    final_dir = output_dir / 'final'
    if final_dir.exists() and not force:
        logger.info('Using existing fine-tuned OnT model: %s', final_dir)
        return str(final_dir)

    from ont import OntologyTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Fine-tuning OnT on %s', owl_path)
    logger.info('Output directory: %s', output_dir)
    OntologyTransformer.fit(
        owl_path=str(owl_path),
        output_dir=str(output_dir),
        num_epochs=int(epochs),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        balanced=bool(balanced),
    )
    return str(final_dir if final_dir.exists() else output_dir)


class OnTModel:
    def __init__(self, model_ref: str):
        self.model_ref = choose_model_dir(model_ref) if Path(str(model_ref)).expanduser().exists() else model_ref
        logger.info('Loading OnT model: %s', self.model_ref)
        try:
            from ont import OntologyTransformer
            self.model = OntologyTransformer.from_pretrained(self.model_ref)
        except ImportError as exc:
            raise ImportError('Install ontology-transformer to load/fine-tune OnT models.') from exc

    def encode_tensor(self, texts: list[str]):
        if hasattr(self.model, 'encode_concept'):
            return self.model.encode_concept(texts)
        try:
            return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        except TypeError:
            return self.model.encode(texts)

    @staticmethod
    def _as_2d_tensor(value):
        import torch
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        elif not hasattr(value, 'dim'):
            value = torch.as_tensor(value)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        return value

    def score_subsumption(self, child_emb, parent_emb) -> float:
        import torch
        child_emb = self._as_2d_tensor(child_emb)
        parent_emb = self._as_2d_tensor(parent_emb)

        if hasattr(self.model, 'manifold'):
            weight = float(getattr(self.model, 'best_lambda', 1.0) or 1.0)
            dist = self.model.manifold.dist(child_emb, parent_emb)
            child_norm = self.model.manifold.dist0(child_emb)
            parent_norm = self.model.manifold.dist0(parent_emb)
            score = -(dist + weight * (parent_norm - child_norm))
            return float(score.item())

        child = child_emb.float()
        parent = parent_emb.float()
        return float(torch.nn.functional.cosine_similarity(child, parent).item())
