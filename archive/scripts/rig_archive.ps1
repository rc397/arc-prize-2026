# Archive dead files (move, not delete) into experiments\archive with a manifest for
# reference. Never touches the active airv_nottt.* run files, the model GGUFs, or the code.
$arch = "D:\arc\experiments\archive"
New-Item -ItemType Directory -Force -Path $arch | Out-Null
$move = @(
    "D:\arc\experiments\ttt_eval_overnight.log",
    "D:\arc\experiments\ttt_eval_overnight.log.err",
    "D:\arc\experiments\ttt_fixed.log",
    "D:\arc\experiments\ttt_fixed.log.err",
    "D:\arc\experiments\ttt_night2.log",
    "D:\arc\experiments\ttt_night2.log.err",
    "D:\arc\experiments\queue.log",
    "D:\arc\experiments\queue.err",
    "D:\arc\experiments\airv_nottt.log.err",
    "D:\arc\experiments\baseline_n3.json",
    "D:\arc\stage0_check.py",
    "D:\arc\prof.py"
)
$manifest = @("Archived 2026-06-20: HF-era TTT/queue logs and Stage 0 scratch, superseded by",
              "the llama.cpp AIRV pipeline. Kept here for reference, safe to delete later.", "")
foreach ($f in $move) {
    if (Test-Path $f) {
        Move-Item $f $arch -Force -ErrorAction SilentlyContinue
        $manifest += ("  " + (Split-Path $f -Leaf) + "   (was " + $f + ")")
    }
}
$manifest -join "`r`n" | Out-File (Join-Path $arch "MANIFEST.txt") -Encoding utf8
$manifest | ForEach-Object { Write-Output $_ }
Write-Output "=== experiments now ==="
Get-ChildItem "D:\arc\experiments" -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("  " + $_.Name) }
