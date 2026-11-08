"""Reconstruct GPU-minutes per config for fig_cost_frontier (B5) from the run logs. Splits the
no-TTT 120 run into the 84 trainable and 36 excluded subsets (the blended pipeline runs TTT on the
84, no-TTT on the 36), and splits the TTT val run into train vs generate."""
import json, re
E = r"D:\arc\experiments"
man = set(json.load(open(E + r"\ttt_manifest_val.json")))


def nottt_split2(path):
    t84 = t36 = prev = 0.0
    for line in open(path):
        m = re.search(r"\[(\d+)/120\]\s+(\w+).*t=([\d.]+)s", line)
        if m:
            tid, t = m.group(2), float(m.group(3))
            dt = t - prev
            prev = t
            if tid in man:
                t84 += dt
            else:
                t36 += dt
    return t84, t36, prev


def ttt_split(path):
    train = run = 0.0
    for line in open(path):
        m = re.search(r"ttt=(\d+)s", line)
        if m:
            train += int(m.group(1))
        m2 = re.search(r"t=([\d.]+)s", line)
        if m2:
            run = max(run, float(m2.group(1)))
    return train, run


t84, t36, tot = nottt_split2(E + r"\airv_nottt.log")
train, run = ttt_split(E + r"\ttt_val.log")
print("nottt  total %.0fs %.1fmin | t84 %.0fs %.1fmin | t36 %.0fs %.1fmin" %
      (tot, tot / 60, t84, t84 / 60, t36, t36 / 60))
print("ttt84  train %.0fs %.1fmin | run %.0fs %.1fmin | gen %.0fs %.1fmin" %
      (train, train / 60, run, run / 60, run - train, (run - train) / 60))
print("blended_time = ttt84_run + nottt_t36 = %.1f min" % ((run + t36) / 60))
