"""Self-contained correctness tests for the model-independent core.
Run: python tests/test_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc import augment as A          # noqa: E402
from arc import eval as E             # noqa: E402
from arc.data import GridSerializer   # noqa: E402


def rand_grid(rng):
    h = int(rng.integers(1, 8)); w = int(rng.integers(1, 8))
    return rng.integers(0, 10, size=(h, w)).tolist()


def test_augment_invertible():
    rng = np.random.default_rng(0)
    for _ in range(500):
        g = rand_grid(rng)
        n = int(rng.integers(1, 5))
        aug = A.random_augmentation(rng, n_examples=n)
        back = A.invert_grid(aug, A.apply_grid(aug, g))
        assert back == g, f"invert failed for d4={aug.d4} color={aug.color}"
    # every D4 op individually, on a non-square grid (catches shape bugs)
    g = [[1, 2, 3], [4, 5, 6]]
    for name in A.D4_NAMES:
        aug = A.Augmentation(d4=name)
        assert A.invert_grid(aug, A.apply_grid(aug, g)) == g, name
    print("OK augment invertible (incl. all D4 on non-square)")


def test_apply_task():
    rng = np.random.default_rng(1)
    task = {"train": [{"input": [[1, 0]], "output": [[0, 1]]},
                      {"input": [[2, 2]], "output": [[3, 3]]}],
            "test":  [{"input": [[5, 6]]}]}
    aug = A.random_augmentation(rng, n_examples=2)
    t2 = A.apply_task(aug, task)
    assert len(t2["train"]) == 2 and "output" in t2["train"][0]
    assert "output" not in t2["test"][0]
    assert A.invert_grid(aug, t2["test"][0]["input"]) == [[5, 6]]
    print("OK apply_task")


def test_serializer():
    s = GridSerializer()
    rng = np.random.default_rng(2)
    for _ in range(200):
        g = rand_grid(rng)
        assert s.decode(s.encode(g)) == g
    s2 = GridSerializer(cell_sep=" ")
    assert s2.decode(s2.encode([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    print("OK serializer round-trip")


def test_scorer():
    sols = {"a": [[[1, 1]]], "b": [[[2]], [[3]]]}   # task b has 2 test outputs
    sub = {
        "a": [{"attempt_1": [[1, 1]], "attempt_2": [[0, 0]]}],   # correct (att1)
        "b": [{"attempt_1": [[9]],    "attempt_2": [[2]]},       # correct (att2)
              {"attempt_1": [[9]],    "attempt_2": [[9]]}],      # wrong
    }
    acc, c, t = E.score_submission(sub, sols)
    assert (c, t) == (2, 3), (c, t)
    assert abs(acc - 2 / 3) < 1e-9
    print("OK scorer (2/3)")


if __name__ == "__main__":
    test_augment_invertible()
    test_apply_task()
    test_serializer()
    test_scorer()
    print("\nALL CORE TESTS PASSED")
