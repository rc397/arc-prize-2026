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
# A KAGGLE_API_TOKEN variable appeared in the rig's environment in early September carrying a
# different token from ~/.kaggle/access_token, and kaggle 2.x prefers the variable, so every call
# here died with "Authentication required". The file is the credential this project manages.
if os.path.isfile(os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")):
    os.environ.pop("KAGGLE_API_TOKEN", None)
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


def push(slug, message, version=0):
    """Resubmit an existing kernel version. The artifact is immutable, so this is a fresh DRAW from
    the same code, which is the point of the replicate series."""
    if listing():
        raise SystemExit("a submission is already dated today in UTC; not spending the slot twice")
    ks = api.kernels_list(mine=True, search=slug.split("/")[-1], page_size=50)
    match = [k for k in ks if getattr(k, "ref", "") == slug]
    if not match:
        raise SystemExit("kernel %s not found among %d of your kernels" % (slug, len(ks)))
    # kernel_version is REQUIRED, and omitting it or passing 0 both return a bare 403 that reads
    # like a permissions failure. It is not one: it is a missing argument. kernels_list reports
    # current_version_number as 0 for this kernel, so fall back to 1, which is the version every
    # replicate has been submitted from.
    ver = version or getattr(match[0], "current_version_number", 0) or 1
    # Version 1 is NOT a safe default. Kaggle retains submittable output only for recent versions,
    # and this kernel reached version 3 while current_version_number still reported 0, so the old
    # hardcoded 1 returned "Did not find provided Notebook Output File" and burned a day working out
    # why. A rejected submission creates nothing and does not spend the daily slot, so walking
    # upward until one is accepted is safe and costs only round trips.
    out = None
    tried = []
    for cand in ([version] if version else [ver + i for i in range(0, 6)]):
        print("trying version %d" % cand, flush=True)
        try:
            out = api.competition_submit_code(file_name="submission.json", message=message,
                                              competition=comp_slug(), kernel=slug,
                                              kernel_version=cand)
            print("accepted at version %d" % cand)
            break
        except Exception as e:
            resp = getattr(e, "response", None)
            body = (resp.text if resp is not None else str(e)) or "(empty body)"
            tried.append((cand, body[:120]))
            if resp is not None and "Notebook Output File" in body:
                continue          # this version has no retained output; the next one may
            raise SystemExit("submit failed HTTP %s: %s"
                             % (getattr(resp, "status_code", "?"), body[:600]))
    if out is None:
        raise SystemExit("no version had retained output; tried %s"
                         % ", ".join("v%d" % c for c, _ in tried))
    print(out)
    print("submitted; the score lands about nine hours later")
    return out


def wait_push(slug, message, version=0):
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
    push(slug, message, version)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        listing(verbose="-v" in sys.argv)
    elif cmd == "api":
        api_surface()
    elif cmd == "push":
        push(sys.argv[2], " ".join(sys.argv[3:]) if len(sys.argv) > 3
             else "replicate for the variance series")
    elif cmd == "wait-push":
        # A submission message is a sentence, and any launcher that does not quote it perfectly
        # delivers it as several argv entries. That is exactly what happened once: "replicate,"
        # arrived where the version number belonged, int() raised, and the waiter died silently
        # before printing a line, so a day's window passed with nothing submitted. Treat everything
        # after the slug as the message, and accept a version only if the LAST argument is a bare
        # integer, which a sentence never is.
        rest = sys.argv[3:]
        ver = 0
        if rest and rest[-1].isdigit():
            ver = int(rest[-1])
            rest = rest[:-1]
        wait_push(sys.argv[2],
                  " ".join(rest) if rest else "replicate for the variance series",
                  ver)
    else:
        raise SystemExit(__doc__)
