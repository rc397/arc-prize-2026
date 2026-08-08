"""Does throughput variation explain the hidden-set variance? A simulation, no GPU, no submissions.

One immutable Kaggle kernel version scored 30.56, 32.64 and 28.47 on three executions. The only
thing that differed was the machine and the day. We argued from that to a throughput account: the
decode is deadline-bounded in three places, so how far the session gets decides where the search
truncates and therefore which candidates ever exist. That is an argument, not a measurement, and the
correlational version is unavailable because Kaggle exposes no runtime on submissions.

It is simulable from data we already own. The measurement run logged per-task wall time and outcome
for 40 tasks. Resample that distribution to a 240-task queue, schedule it over 4 workers against the
real 42480-second wall, vary the machine speed, and read off the resulting score distribution. Then
compare its standard deviation against the 1.35 points measured across six runs of the unchanged
configuration.

  If the simulation reproduces roughly 1.35, throughput explains the variance quantitatively.
  If it produces much less, the residue belongs to decoder nondeterminism, which we measured
  directly at 12.5 percent same-seed output disagreement and 0.543 same-config pool Jaccard.

The scheduling model is deliberately the simple one the pipeline actually uses: tasks are taken off a
shared queue in order, a worker takes the next task when it frees, and a task that has not STARTED
before the wall scores zero. A task already running when the wall arrives is assumed to finish, which
is the generous reading and biases the simulation against finding an effect.

Two things it cannot capture and we do not claim: per-task truncation inside the cap (a task that
starts late still gets its full decode here), and decoder nondeterminism, which is the other
mechanism entirely. So this measures the tail-dropping component of throughput variance alone, and
its answer is therefore a LOWER bound on what throughput explains.

Usage: python sim_queue_variance.py [TRIALS=4000]
"""
import csv
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "paper", "data", "task_timing.csv")

N_TASKS = 240
N_WORKERS = 4
WALL = 42480.0          # 12 hours less the 600-second reserve the notebook keeps
CREDIT = 100.0 / 240.0  # points per solved task
MEASURED_SD = 1.35      # six runs of the unchanged configuration
MEASURED_RANGE = 4.17


def load():
    rows = list(csv.DictReader(open(CSV)))
    return [(float(r["wall_seconds"]), r["outcome"] in ("solved",)) for r in rows]


def make_queue(pool, rng):
    """The 240 hidden tasks are the SAME every run, so the queue is drawn once and reused.

    An earlier version resampled it per trial and reported sd 2.87 at zero speed variation, which is
    the sampling noise of the simulation and not a property of the pipeline. Only the machine varies
    between real runs, so only the machine may vary between trials.
    """
    q = [pool[rng.randrange(len(pool))] for _ in range(N_TASKS)]
    rng.shuffle(q)
    return q


def one_run(tasks, speed, rng):
    """Schedule a fixed queue over workers at a given machine speed, and score it."""
    free = [0.0] * N_WORKERS
    solved = 0
    started = 0
    for dur, ok in tasks:
        w = min(range(N_WORKERS), key=lambda i: free[i])
        if free[w] >= WALL:
            continue                      # never started, scores zero
        started += 1
        free[w] += dur / speed
        if ok:
            solved += 1
    return solved * CREDIT, started


def main(trials):
    pool = load()
    base_rate = sum(1 for _, ok in pool if ok) / len(pool)
    mean_dur = st.mean(d for d, _ in pool)
    print("measured pool: %d tasks, mean %.0fs, solved rate %.3f"
          % (len(pool), mean_dur, base_rate))
    print("queue: %d tasks over %d workers against a %.0fs wall" % (N_TASKS, N_WORKERS, WALL))
    print("nominal load: %.0fs per worker at speed 1.0\n"
          % (N_TASKS * mean_dur / N_WORKERS))

    rng = random.Random(3)
    queue = make_queue(pool, rng)
    # Speed spread is the unknown. Report several so the reader can see the sensitivity rather than
    # take one assumed number on trust. The queue is fixed; only the machine varies.
    for spread in (0.00, 0.05, 0.10, 0.20, 0.35):
        scores, starts = [], []
        for _ in range(trials):
            speed = rng.gauss(1.0, spread) if spread else 1.0
            speed = max(0.3, speed)
            s, started = one_run(queue, speed, rng)
            scores.append(s)
            starts.append(started)
        sd = st.pstdev(scores)
        rng_pts = max(scores) - min(scores)
        print("speed sd %4.0f%%  ->  score mean %5.2f  sd %5.2f  range %5.2f  tasks started %5.1f/%d"
              % (spread * 100, st.mean(scores), sd, rng_pts, st.mean(starts), N_TASKS))

    print("\nmeasured on the hidden set: sd %.2f, range %.2f over six runs"
          % (MEASURED_SD, MEASURED_RANGE))
    print("\nVERDICT")
    # re-run the middle setting to judge against
    scores = []
    for _ in range(trials):
        speed = max(0.3, rng.gauss(1.0, 0.10))
        scores.append(one_run(queue, speed, rng)[0])
    sd10 = st.pstdev(scores)
    started = st.mean([one_run(queue, 1.0, rng)[1] for _ in range(50)])
    if started >= N_TASKS - 0.5:
        print("  At the measured task durations the queue FINISHES with time to spare: %.1f of %d"
              % (started, N_TASKS))
        print("  tasks start even at speed 1.0. Tail-dropping therefore cannot be the mechanism, and")
        print("  the variance must come from truncation inside the per-task cap or from decoder")
        print("  nondeterminism, not from tasks never being reached. That REFUTES the simple")
        print("  throughput story we told and narrows it to a different one.")
    elif sd10 >= 0.6 * MEASURED_SD:
        print("  Tail-dropping alone reproduces sd %.2f against the measured %.2f, so throughput"
              % (sd10, MEASURED_SD))
        print("  explains most of the hidden-set variance and the account stands quantitatively.")
    else:
        print("  Tail-dropping alone gives sd %.2f against the measured %.2f, so it is real but"
              % (sd10, MEASURED_SD))
        print("  cannot be the whole story. The residue belongs to per-task truncation and to")
        print("  decoder nondeterminism, measured separately at 12.5%% same-seed disagreement.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000) else 1)
