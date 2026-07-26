"""Pull the live leaderboard and convert points into places.

The board is plateaued on public-notebook forks, so the useful question is not how many points a lever is
worth but how many entries sit between us and the next thin stretch. This prints the density directly.

Runs on the rig, where the Kaggle token lives. Usage: python rig_leaderboard.py [SLUG] [OUR_SCORE]
"""
import json
import os
import sys
from collections import Counter

SLUG = sys.argv[1] if len(sys.argv) > 1 else "arc-prize-2026-arc-agi-2"
OURS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.97
OUT = r"D:\arc\experiments\leaderboard.json"

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

def fetch_all(slug, page_size=200):
    """The endpoint pages, and a single page hides exactly the density we are trying to measure."""
    out, token, seen = [], None, 0
    while True:
        page = api.competition_leaderboard_view(slug, page_size=page_size, page_token=token)
        if not page:
            break
        out.extend(page)
        token = getattr(api, "next_page_token", None) or getattr(page, "next_page_token", None)
        if not token or len(out) == seen:
            break
        seen = len(out)
    return out


rows = None
for slug in [SLUG, "arc-prize-2026", "arc-prize-2026-arc-agi-2"]:
    try:
        rows = fetch_all(slug)
        if rows:
            SLUG = slug
            break
    except Exception as e:
        print("slug %s failed, %s" % (slug, str(e)[:90]))

if not rows:
    sys.exit("no leaderboard returned")


def score_of(r):
    v = getattr(r, "score", None) or getattr(r, "publicScore", None)
    return float(v) if v not in (None, "") else None


scores = sorted([s for s in (score_of(r) for r in rows) if s is not None], reverse=True)
print("competition %s, %d scored entries" % (SLUG, len(scores)))

with open(OUT, "w") as f:
    json.dump({"slug": SLUG, "scores": scores}, f)

# where do we sit, and how many share our exact number
better = sum(1 for s in scores if s > OURS)
same = sum(1 for s in scores if abs(s - OURS) < 1e-9)
print("our score %.2f, %d entries above, %d sharing it exactly" % (OURS, better, same))

print("\nscore needed to reach each rank")
for rank in (1, 3, 5, 10, 20, 30, 50, 100, 150, 200):
    if rank <= len(scores):
        print("  top %-4d needs %6.2f  (%+.2f from here)" % (rank, scores[rank - 1], scores[rank - 1] - OURS))

print("\nentries per score band above us")
bands = Counter()
for s in scores:
    if s >= OURS:
        bands[int(s)] += 1
for lo in sorted(bands, reverse=True):
    print("  %2d to %2d  %4d entries  %s" % (lo, lo + 1, bands[lo], "#" * min(bands[lo], 60)))

print("\nthe ten most common exact scores, the fork clusters")
for s, n in Counter(scores).most_common(10):
    print("  %6.2f  %4d entries" % (s, n))
