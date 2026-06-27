#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from ont_model import finetune_ont


def main():
    parser = argparse.ArgumentParser(description='Fine-tune an Ontology Transformer model on an OWL ontology')
    parser.add_argument('--owl', required=True)
    parser.add_argument('--output', default='runs/ont_finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    model_dir = finetune_ont(
        owl_path=args.owl,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        balanced=args.balanced,
        force=args.force,
    )
    print(model_dir)


if __name__ == '__main__':
    main()
