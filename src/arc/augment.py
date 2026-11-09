"""Invertible ARC augmentations: D4 geometry, color permutation, example reorder.

apply_grid maps a grid into the augmented frame; invert_grid maps a prediction back.
Geometry and recoloring commute, so the inverse undoes both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Grid = List[List[int]]


# D4 geometric ops and their inverses, keyed by name
def _id(a):        return a
def _rot90(a):     return np.rot90(a, 1)            # counter-clockwise
def _rot180(a):    return np.rot90(a, 2)
def _rot270(a):    return np.rot90(a, 3)
def _flip_lr(a):   return np.fliplr(a)
def _flip_ud(a):   return np.flipud(a)
def _transpose(a): return a.T
def _anti(a):      return np.fliplr(np.fliplr(a).T)  # anti-diagonal transpose (involution)

D4_FWD = {
    "identity": _id, "rot90": _rot90, "rot180": _rot180, "rot270": _rot270,
    "flip_lr": _flip_lr, "flip_ud": _flip_ud, "transpose": _transpose, "anti": _anti,
}
D4_INV = {
    "identity": _id, "rot90": _rot270, "rot180": _rot180, "rot270": _rot90,
    "flip_lr": _flip_lr, "flip_ud": _flip_ud, "transpose": _transpose, "anti": _anti,
}
D4_NAMES = list(D4_FWD)


@dataclass(frozen=True)
class Augmentation:
    d4: str = "identity"
    # color[c] = relabeled color; default identity permutation of 0..9
    color: Tuple[int, ...] = tuple(range(10))
    # order in which train examples are presented (indices into the train list)
    example_order: Optional[Tuple[int, ...]] = None

    def inv_color(self) -> Tuple[int, ...]:
        inv = [0] * len(self.color)
        for c, nc in enumerate(self.color):
            inv[nc] = c
        return tuple(inv)


def _recolor(arr: np.ndarray, mapping: Sequence[int]) -> np.ndarray:
    return np.asarray(mapping, dtype=arr.dtype)[arr]


def apply_grid(aug: Augmentation, grid: Grid) -> Grid:
    a = np.asarray(grid, dtype=np.int64)
    a = D4_FWD[aug.d4](a)        # geometry
    a = _recolor(a, aug.color)   # recolor
    return a.tolist()


def invert_grid(aug: Augmentation, grid: Grid) -> Grid:
    """Map a prediction from the augmented frame back to the original frame."""
    a = np.asarray(grid, dtype=np.int64)
    a = _recolor(a, aug.inv_color())  # undo recolor
    a = D4_INV[aug.d4](a)             # undo geometry
    return a.tolist()


def apply_task(aug: Augmentation, task: dict) -> dict:
    """Transform every grid in a task and reorder train examples.
    Test outputs (if present) are transformed too; absent ones are left out."""
    def pair(p):
        q = {"input": apply_grid(aug, p["input"])}
        if "output" in p:
            q["output"] = apply_grid(aug, p["output"])
        return q

    train = [pair(p) for p in task["train"]]
    if aug.example_order is not None:
        train = [train[i] for i in aug.example_order]
    test = [pair(p) for p in task["test"]]
    return {"train": train, "test": test}


def random_augmentation(rng: np.random.Generator, *, use_d4: bool = True,
                        recolor: bool = True, keep_background: bool = False,
                        n_examples: Optional[int] = None,
                        reorder: bool = True) -> Augmentation:
    d4 = str(rng.choice(D4_NAMES)) if use_d4 else "identity"
    if recolor:
        if keep_background:
            perm = np.arange(10)
            perm[1:] = rng.permutation(np.arange(1, 10))
            color = tuple(int(x) for x in perm)
        else:
            color = tuple(int(x) for x in rng.permutation(10))
    else:
        color = tuple(range(10))
    order = None
    if reorder and n_examples:
        order = tuple(int(x) for x in rng.permutation(n_examples))
    return Augmentation(d4=d4, color=color, example_order=order)
