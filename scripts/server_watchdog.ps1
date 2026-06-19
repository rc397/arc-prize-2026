# Keeps llama-server alive: if the process is gone, restart it within ~15s. Runs until
# killed (Stop-Process) or a 3-hour safety cap. No log dependency by design.
# Config rationale: single stream (-np 1) avoids the concurrent-generation VRAM spike that
# crashed -np 2/-np 4; prompt/idle-slot cache disabled (every ARC prompt is unique, so the
# cache was pure overhead and a crash source); -c 16384 fits the largest ARC prompts.
$exe = "D:\arc\llamacpp\llama-server.exe"
$svArgs = @('-m','D:\arc\models\arc-qwen4b-q8_0.gguf','-ngl','99','-c','16384','-np','1',
            '--host','127.0.0.1','--port','8080','--no-webui',
            '--no-cache-prompt','--no-cache-idle-slots')
for ($n = 0; $n -lt 720; $n++) {
    if (-not (Get-Process llama-server -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $exe -ArgumentList $svArgs -WindowStyle Hidden `
            -RedirectStandardOutput "D:\arc\srv.log" -RedirectStandardError "D:\arc\srv.err"
    }
    Start-Sleep -Seconds 15
}
