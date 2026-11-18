# Reproduz o estudo completo de um dataset do zero (etapas retomáveis).
# Uso: .\scripts\run_all.ps1 [-Dataset eurosat|stl10] [-Epochs N] [-NJobs 4]
param([string]$Dataset = "eurosat", [int]$Epochs = 0, [int]$NJobs = 4)
$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$stages = "prepare", "teacher", "benchmark", "tune", "train-all", "evaluate", "report"
$first = $true
foreach ($s in $stages) {
    Write-Host ">>> etapa: $s ($Dataset)"
    $cmdArgs = @("-m", "mo433_projeto_final", $s, "--dataset", $Dataset, "--n-jobs", $NJobs)
    if ($first -and $Epochs -gt 0) { $cmdArgs += @("--epochs", $Epochs) }
    & $py @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "etapa $s falhou (código $LASTEXITCODE)" }
    $first = $false
}
Write-Host "Resultados em results\$Dataset\ (summary.md, tabelas CSV e figuras PNG)."
