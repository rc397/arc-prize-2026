# Remove build/debug junk no longer needed. Leaves model GGUFs, code, data, and the
# in-flight eval/server files untouched.
$junk = @(
    "D:\arc\llamacpp\llama.zip",
    "D:\arc\llamacpp\cudart.zip",
    "D:\arc\load.log",
    "D:\arc\cli.log",
    "D:\arc\patch_vocab.py",
    "D:\arc\patch_merges.py"
)
$freed = 0.0
foreach ($f in $junk) {
    if (Test-Path $f) {
        $mb = (Get-Item $f).Length / 1MB
        $freed += $mb
        Remove-Item $f -Force -ErrorAction SilentlyContinue
        Write-Output ("deleted {0} ({1:N1} MB)" -f $f, $mb)
    }
}
Write-Output ("freed {0:N0} MB" -f $freed)
Write-Output "=== D:\arc top level ==="
Get-ChildItem "D:\arc" -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("  " + $_.Name) }
Write-Output "=== models\*.gguf ==="
Get-ChildItem "D:\arc\models\*.gguf" -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("  {0}  {1:N2} GB" -f $_.Name, ($_.Length / 1GB)) }
Write-Output "=== experiments ==="
Get-ChildItem "D:\arc\experiments" -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("  " + $_.Name) }
