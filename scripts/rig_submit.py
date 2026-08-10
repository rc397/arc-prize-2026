"""List submissions, and resubmit an existing kernel version to the code competition.

The replicate series is the project's variance instrument: the SAME immutable kernel version, run
again, so every difference between runs is the competition environment rather than our code. It has
produced 30.56, 32.64, 28.47, 31.39 from one artifact, which is the single most important number in
the paper and needs more draws.

Submitting is rate limited to one a day on a UTC boundary, so this always lists first and refuses to
push if the day already has one, rather than discovering that from an error.

The token lives on the rig at ~/.kaggle/access_token and is never printed.

Usage:
  python rig_submit.py list                       recent submissions, newest first
  python rig_submit.py api                        what this kaggle build exposes for submitting
  python rig_submit.py push <kernel-slug> [msg]   resubmit that kernel's latest version
"""
import datetime
import os
import sys

COMP = "arc-prize-2026"
from kaggle.api.kaggle_api_extended import KaggleApi          # noqa: E402

api = KaggleApi()
api.authenticate()


def comp_slug():
    for s in (COMP, "arc-prize-2026-arc-agi-2"):
        try:
            api.competition_submissions(s)
            return s
        except Exception:
            continue
    raise SystemExit("no competition slug accepted by this account")


def score_of(x):
    """The score field has been named three different things across kaggle builds."""
    for n in ("publicScore", "public_score", "publicScoreNullable", "score"):
        v = getattr(x, n, None)
        if v not in (None, ""):
            return v
    return None


def listing(verbose=False):
    s = comp_slug()
    subs = api.competition_submissions(s)
    print("competition: %s, %d submissions" % (s, len(subs)))
    today = datetime.datetime.now(datetime.timezone.utc).date()
    used = 0
    for x in subs[:12]:
        d = getattr(x, "date", None)
        when = d.strftime("%Y-%m-%d %H:%M UTC") if hasattr(d, "strftime") else str(d)
        if hasattr(d, "date") and d.date() == today:
            used += 1
        print("  %-20s %-22s %-12s %-8s  %s"
              % (getattr(x, "ref", "?"), when, str(getattr(x, "status", "?")).split(".")[-1],
                 score_of(x), (getattr(x, "description", "") or "")[:44]))
    if verbose and subs:
        print("")
        print("fields on a submission object:")
        for k in sorted(vars(subs[0]) if hasattr(subs[0], "__dict__") else []):
            print("  %-28s %r" % (k, getattr(subs[0], k)))
    print("")
    print("submissions dated today (%s UTC): %d" % (today, used))
    return used


def api_surface():
    names = [n for n in dir(api) if "submit" in n.lower() or "kernel" in n.lower()]
    for n in sorted(names):
        print("  %s" % n)


def push(slug, message):
    """Resubmit an existing kernel version. The artifact is immutable, so this is a fresh DRAW from
    the same code, which is the point of the replicate series."""
    if listing():
        raise SystemExit("a submission is already dated today in UTC; not spending the slot twice")
    ks = api.kernels_list(mine=True, search=slug.split("/")[-1], page_size=50)
    match = [k for k in ks if getattr(k, "ref", "") == slug]
    if not match:
        raise SystemExit("kernel %s not found among %d of your kernels" % (slug, len(ks)))
    print("found %s, last run %s" % (slug, getattr(match[0], "lastRunTime", "?")))
    out = api.competition_submit_code(file_name="submission.json", message=message,
                                      competition=comp_slug(), kernel=slug)
    print(out)
    print("submitted; the score lands about nine hours later")
    return out


def wait_push(slug, message):
    """Block until the UTC day rolls, then submit once.

    The quota is one a day on a UTC boundary and today's is already spent, so 'push it' means push
    at the first legal instant rather than not at all. It re-checks the listing after waking, so if
    a submission appears in the meantime this does nothing rather than racing it.
    """
    import time
    start = datetime.datetime.now(datetime.timezone.utc).date()
    print("waiting for the UTC day to roll past %s" % start, flush=True)
    while datetime.datetime.now(datetime.timezone.utc).date() <= start:
        time.sleep(60)
    print("UTC day is now %s" % datetime.datetime.now(datetime.timezone.utc).date(), flush=True)
    push(slug, message)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        listing(verbose="-v" in sys.argv)
    elif cmd == "api":
        api_surface()
    elif cmd == "push":
        push(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "replicate for the variance series")
    elif cmd == "wait-push":
        wait_push(sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else "replicate for the variance series")
    else:
        raise SystemExit(__doc__)
