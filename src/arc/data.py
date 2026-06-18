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
