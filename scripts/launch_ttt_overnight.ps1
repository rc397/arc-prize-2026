# Launch the full 120-task TTT + AIRV run, detached so it survives the laptop closing.
# Run this on the rig tonight:
#   ssh rig "powershell -NoProfile -ExecutionPolicy Bypass -File D:\arc\scripts\launch_ttt_overnight.ps1"
# The run writes its submission incrementally (crash-safe) and prints a final SCORE line.
$ProgressPreference = "SilentlyContinue"
$env:PYTHONIOENCODING = "utf-8"
Get-Process python, python3.13, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
$job = Start-Process -FilePath python -ArgumentList 'D:\arc\scripts\run_ttt_airv.py', '0', '40' `
    -WindowStyle Hidden -WorkingDirectory "D:\arc" `
    -RedirectStandardOutput "D:\arc\experiments\airv_ttt.log" `
    -RedirectStandardError "D:\arc\experiments\airv_ttt.err" -PassThru
Write-Output ("TTT overnight run launched, PID=" + $job.Id)
Write-Output "Monitor:  Get-Content D:\arc\experiments\airv_ttt.log -Tail 6"
Write-Output "Done when the log ends with a line 'SCORE acc=...'; submission at experiments\airv_ttt_eval.json"
