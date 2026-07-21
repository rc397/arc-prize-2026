# Archived notebooks

Builds that were assembled but deliberately never submitted. They are kept so the reasoning stays
attached to the artifact rather than living only in a commit message.

## arc_submission_v7b.ipynb

The v6 deep-TTT recipe (18.19 hidden) with the LoRA rank pushed from 32 to 128, built on an early
partial reading that higher rank helped the greedy vote.

**Not used, and it should not be submitted.** The full rank sweep afterwards scored 24, 27, 28 and 24
of 125 for ranks 16, 32, 64 and 128 at 60 steps. That is an inverted U peaking at the rank v6.1 already
ships, with rank 128 joint worst, so this build would be expected to land below the 18.19 it forked
from. It was retired before a submission slot was spent on it.

Rank 64 leads rank 32 by a single output out of 125, which is inside seed noise at double the adapter
cost, so no point on that curve justifies a resubmission either. The rank lever is closed unless a
future sweep at a different training context moves the peak.

Builder is scripts/build_v7b_from_v6.py, which carries the same warning.
