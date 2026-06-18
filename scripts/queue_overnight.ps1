param([int]$WaitPid = 0)
# Sequential overnight job queue for the rig. Waits for an in-progress run (by pid) to finish,
# then runs TTT variations one after another, each writing its own log under experiments/.
# Launch detached so it survives the laptop closing:
#   Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass",
#     "-File","D:\arc\scripts\queue_overnight.ps1","-WaitPid","<pid>") -WindowStyle Hidden -PassThru
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$md  = "D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
$ch  = "D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
$sol = "D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

function Run-Job($name, $extra) {
    $out = "D:\arc\experiments\$name.json"
    $log = "D:\arc\experiments\$name.log"
    $a = @("D:\arc\scripts\run_ttt.py", "--model-dir", $md, "--challenges", $ch,
           "--solutions", $sol, "--out", $out) + $extra
    & python @a *> $log
}

if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}

Run-Job "ttt_naug16_sample_n40" @("--n", "40", "--n-aug", "16", "--do-sample", "--temperature", "0.9")
Run-Job "ttt_steps250_n40"      @("--n", "40", "--steps", "250")
Run-Job "ttt_rank64_n40"        @("--n", "40", "--rank", "64")
