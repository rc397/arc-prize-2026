"""ARC task IO and grid/text serialization.

GridSerializer defaults (no cell separator, newline row separator) match the NVARC
model's grid format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

Grid = List[List[int]]


def load_json(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_challenges(path) -> Dict:
    """task_id -> {"train": [...], "test": [...]}."""
    return load_json(path)


def load_solutions(path) -> Dict:
    """task_id -> [grid, ...] (one per test input)."""
    return load_json(path)


@dataclass
class GridSerializer:
    cell_sep: str = ""     # between cells in a row ("" => one char per cell)
    row_sep: str = "\n"    # between rows

    def encode(self, grid: Grid) -> str:
        return self.row_sep.join(
            self.cell_sep.join(str(int(v)) for v in row) for row in grid
        )

    def decode(self, text: str) -> Grid:
        rows = [r for r in text.split(self.row_sep) if r != ""]
        out: Grid = []
        for r in rows:
            cells = list(r) if self.cell_sep == "" else r.split(self.cell_sep)
            out.append([int(c) for c in cells if str(c).strip() != ""])
        return out


def load_consolidated(challenges_path, solutions_path=None, n=None):
    """Official Kaggle format: challenges[tid] = {train, test}, solutions[tid] = [grid]."""
    ch = load_json(challenges_path)
    sol = load_json(solutions_path) if solutions_path else {}
    if n:
        ids = list(ch)[:n]
        ch = {k: ch[k] for k in ids}
        sol = {k: sol[k] for k in ids if k in sol}
    return ch, sol


def write_submission(submission, challenges, path):
    """Emit submission.json with every task_id and both attempts present per test input."""
    out = {}
    for tid, task in challenges.items():
        preds = submission.get(tid, [])
        entry = []
        for i in range(len(task["test"])):
            p = preds[i] if i < len(preds) else {}
            a1 = p.get("attempt_1") or [[0]]
            a2 = p.get("attempt_2") or a1
            entry.append({"attempt_1": a1, "attempt_2": a2})
        out[tid] = entry
    Path(path).write_text(json.dumps(out))
    return out
