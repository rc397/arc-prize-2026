"""Competition scoring for ARC-AGI-2.

Metric: for each test output, score 1 if either of the 2 attempts matches the
ground-truth grid exactly, else 0; final score = total correct / total test
outputs. Mirrors the Kaggle ARC-AGI-2 evaluation.
"""
from __future__ import annotations

from typing import Dict, List

Grid = List[List[int]]


def normalize(grid) -> Grid:
    return [[int(v) for v in row] for row in grid]


def grids_equal(a, b) -> bool:
    return normalize(a) == normalize(b)


def score_submission(submission: Dict, solutions: Dict):
    """submission[tid] = [ {attempt_1, attempt_2}, ... ] per test input.
    solutions[tid]   = [ grid, ... ] ground-truth per test input.
    Returns (accuracy, n_correct, n_total).
    """
    correct = total = 0
    for tid, gts in solutions.items():
        preds = submission.get(tid, [])
        for i, gt in enumerate(gts):
            total += 1
            gt_n = normalize(gt)
            if i < len(preds):
                p = preds[i] or {}
                if any(k in p and normalize(p[k]) == gt_n
                       for k in ("attempt_1", "attempt_2")):
                    correct += 1
    acc = correct / total if total else 0.0
    return acc, correct, total
