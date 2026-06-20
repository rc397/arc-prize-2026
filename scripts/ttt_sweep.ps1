# Overnight TTT method sweep: three configs on a 15-task subset, each scored against no-TTT.
# Configs target the overfit hypothesis (the base adapter hits loss 0.000 in 40 steps): fewer
# steps, lower rank, and dropout. Each config is independent; a failure in one does not stop
# the others. Run detached.
$env:PYTHONIOENCODING = "utf-8"
$N = 15
# "steps rank dropout lr tag"
$configs = @(
    "40 32 0.0  1e-4 base40",
    "20 16 0.1  1e-4 s20r16d",
    "8  8  0.05 1e-4 s8r8d"
)
foreach ($c in $configs) {
    $p = $c -split '\s+'
    Write-Output ("=== CONFIG " + $p[4] + " (steps=" + $p[0] + " rank=" + $p[1] + " drop=" + $p[2] + ") ===")
    python D:\arc\scripts\ttt_phase1.py $N $p[0] $p[1] $p[2] $p[3] $p[4]
    python D:\arc\scripts\ttt_phase1b.py $p[4]
    python D:\arc\scripts\ttt_phase2.py 30 1.0 $p[4]
}
Write-Output "SWEEP DONE"
