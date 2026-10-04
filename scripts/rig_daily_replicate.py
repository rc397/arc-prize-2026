"""Resubmit the replicate kernel once per UTC day until 1 November, extending the variance series.

The paper's central measurement is twelve executions of one immutable Kaggle kernel version,
339772104. After the season closed the daily slots went unused, and each resubmission is a free
further draw from the same artifact, so this spends them on the series. It submits shortly after
each UTC midnight, leaves a day alone if its slot is already used, and stops after 1 November or as
soon as the stop file exists. If Kaggle no longer holds that version's output it falls back to the
fresh build of the same configuration, 346281718, which the paper pools with the replicate
(variance ratio 1.01), and says so in the submission description.

Stop it early by creating D:/arc/experiments/STOP_DAILY_REPLICATE.
"""
import datetime
import os
import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_submit  # noqa: E402

SOURCES = [
    ("rc7530/v7-2-replicate", "339772104",
     "v7.2 replicate, kernel version 339772104, draw %d of the series. Nothing changes between runs; "
     "each is a further draw from the same artifact for the paper's variance measurement."),
    ("rc7530/arc-agi-2-v7-2-rebuild-control", "346281718",
     "Fresh build of the v7.2 configuration, kernel version 346281718, draw %d. The replicate's "
     "retained output had expired, so the daily series continues on this build, which the paper "
     "pools with the replicate."),
]
LAST_DAY = datetime.date(2026, 11, 1)
STOP = os.path.join(os.path.dirname(HERE), "experiments", "STOP_DAILY_REPLICATE")


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def log(msg):
    print("%s  %s" % (now().strftime("%Y-%m-%d %H:%M UTC"), msg), flush=True)


def draws_of(version):
    subs = rig_submit.api.competition_submissions(rig_submit.comp_slug(), page_size=100)
    return sum(1 for x in subs if ("scriptVersionId=" + version) in (getattr(x, "url", "") or ""))


def submit_today():
    """True once today's slot holds a submission, whether ours or an earlier one."""
    for slug, version, template in SOURCES:
        try:
            rig_submit.push(slug, template % (draws_of(version) + 1))
            log("submitted from %s" % slug)
            return True
        except SystemExit as e:
            text = str(e)
            if "already dated today" in text:
                log("today's slot is already used; leaving it")
                return True
            log("%s did not go through: %s" % (slug, text[:300]))
    return False


def main():
    log("daily replicate loop started, last day %s, stop file %s" % (LAST_DAY, STOP))
    while now().date() <= LAST_DAY:
        if os.path.exists(STOP):
            log("stop file found; exiting")
            return
        day = now().date()
        try:
            done = submit_today()
        except Exception:
            log("raised, retrying in ten minutes:\n%s" % traceback.format_exc()[-800:])
            time.sleep(600)
            continue
        if not done:
            log("nothing accepted for %s; retrying in an hour" % day)
            time.sleep(3600)
            continue
        while now().date() == day and not os.path.exists(STOP):
            time.sleep(600)
        time.sleep(300)   # land a few minutes after midnight, clear of the boundary
    log("past %s; the series loop is finished" % LAST_DAY)


if __name__ == "__main__":
    main()
