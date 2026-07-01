"""White-box plausibility filters for ARC candidate outputs.

From the T5-ARC recipe that took a 200M model to 27% on ARC-AGI-2 semi-private (arxiv 2603.06590):
discard candidates that violate the task's colour palette or the train pairs' size pattern before the
vote. Filtering removes noise and sharpens the vote, unlike adding more candidates, which dilutes it.

Rule-based and cheap, no model. Reused by the rig harness and the submission notebook.
"""


def _dims(g):
    return (len(g), len(g[0]) if g and g[0] else 0)


def _colours(g):
    return {v for row in g for v in row}


def size_rule(task):
    """Infer the expected test-output size from the train pairs. Returns "same" if every output matches
    its input size, a constant (h, w) if all outputs share one size, or None if the relation is variable
    and cannot be filtered on."""
    pairs = task["train"]
    if all(_dims(p["output"]) == _dims(p["input"]) for p in pairs):
        return "same"
    outs = {_dims(p["output"]) for p in pairs}
    if len(outs) == 1:
        return next(iter(outs))
    return None


def palette(task):
    cols = set()
    for p in task["train"]:
        cols |= _colours(p["input"]) | _colours(p["output"])
    return cols


def plausible(grid, task, test_input):
    """True if the candidate output is consistent with the task's colours and size pattern."""
    if not grid or not grid[0]:
        return False
    # colour filter: an output colour outside the task palette (plus the test input's own colours) is
    # almost never a real ARC output.
    if not _colours(grid) <= (palette(task) | _colours(test_input)):
        return False
    # size filter: enforce the train pairs' size relation when it is consistent.
    rule = size_rule(task)
    if rule == "same" and _dims(grid) != _dims(test_input):
        return False
    if isinstance(rule, tuple) and _dims(grid) != rule:
        return False
    return True
