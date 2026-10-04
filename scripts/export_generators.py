"""Export the verified ARC-AGI-2 generator corpus for release.

Reads every experiments/genrules_*.jsonl, keeps one program per task that reproduced all of its
task's demonstration pairs under execution, preferring the main no-reasoning pass and then any
program that also defines a sampler. Each kept program is re-run here against its demonstrations as
a sanity check and against the task's held-out test outputs, which no program saw during generation.
Writes corpus/arc2_generators.jsonl, one record per task, and corpus/arc2_generators_summary.json.

The task grids are not copied: each record names its task ID in the public ARC-AGI-2 training split.
"""
import collections
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyexec  # noqa: E402

BASE = os.path.dirname(HERE)
EXP = os.path.join(BASE, "experiments")
OUT = os.path.join(BASE, "corpus")

# preference order when one task was verified by more than one condition
CONDITIONS = [
    ("genrules_nothink.jsonl", "deepseek-v4-flash", "reasoning off"),
    ("genrules_effort_low.jsonl", "deepseek-v4-flash", "reasoning effort low"),
    ("genrules_retryctl.jsonl", "deepseek-v4-flash", "reasoning off, blind retry"),
    ("genrules_repair.jsonl", "deepseek-v4-flash", "reasoning off, repair from traceback"),
    ("genrules_maxeffort.jsonl", "deepseek-v4-flash", "reasoning effort max"),
    ("genrules_qwen_local.jsonl", "qwen3.8-27b-q3_k_m", "local, reasoning off"),
]


def find_data():
    for p in glob.glob(os.path.join(BASE, "**", "arc-agi_training_challenges.json"), recursive=True):
        if "models" in p.split(os.sep):
            continue
        sol = os.path.join(os.path.dirname(p), "arc-agi_training_solutions.json")
        if os.path.isfile(sol):
            return p, sol
    raise SystemExit("ARC-AGI-2 training challenges and solutions not found under %s" % BASE)


def main():
    chal_path, sol_path = find_data()
    challenges = json.load(open(chal_path, encoding="utf-8"))
    solutions = json.load(open(sol_path, encoding="utf-8"))
    print("data: %s (%d tasks)" % (chal_path, len(challenges)))

    chosen = {}
    for rank, (fname, model, setting) in enumerate(CONDITIONS):
        path = os.path.join(EXP, fname)
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") is not True or not r.get("src"):
                continue
            task = str(r["task"])
            cand = (rank, r, model, setting)
            prev = chosen.get(task)
            # a sampler outranks the condition order: a generator without one cannot mint data
            if prev is None or (bool(r.get("sampler")) and not bool(prev[1].get("sampler"))):
                chosen[task] = cand
    print("%d tasks with a verified program" % len(chosen))

    records = []
    for task in sorted(chosen):
        rank, r, model, setting = chosen[task]
        t = challenges[task]
        demo_in = [p["input"] for p in t["train"]]
        demo_out = [p["output"] for p in t["train"]]
        test_in = [p["input"] for p in t["test"]]
        test_out = solutions[task]
        d = pyexec.run_candidate(r["src"], demo_in, timeout=8.0)
        demos_ok = d is not None and len(d) == len(demo_out) and all(a == b for a, b in zip(d, demo_out))
        o = pyexec.run_candidate(r["src"], test_in, timeout=8.0)
        test_ok = o is not None and len(o) == len(test_out) and all(a == b for a, b in zip(o, test_out))
        records.append({
            "task": task,
            "model": model,
            "setting": setting,
            "has_sampler": bool(r.get("sampler")),
            "demo_pairs": len(demo_out),
            "test_pairs": len(test_out),
            "demos_reproduced": demos_ok,
            "test_outputs_reproduced": test_ok,
            "output_tokens": int(r.get("tok_out") or 0),
            "program": r["src"],
        })
        print("  %s  demos %-5s test %-5s sampler %-5s %s"
              % (task, demos_ok, test_ok, bool(r.get("sampler")), setting), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "arc2_generators.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "generators": len(records),
        "with_sampler": sum(r["has_sampler"] for r in records),
        "demos_reproduced_on_recheck": sum(r["demos_reproduced"] for r in records),
        "test_outputs_reproduced": sum(r["test_outputs_reproduced"] for r in records),
        "by_setting": dict(collections.Counter(r["setting"] for r in records)),
        "training_tasks_in_split": len(challenges),
    }
    json.dump(summary, open(os.path.join(OUT, "arc2_generators_summary.json"), "w",
                            encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
