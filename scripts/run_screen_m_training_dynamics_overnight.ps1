param(
    [string]$ConfigPath = "configs/benchmarks/mc_maze_stndt_lite_diverse_screen_m_training_dynamics.yaml",
    [string]$VenvRoot = "$HOME\.venvs\nlb-project",
    [int]$TimeoutSeconds = 28800,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunName = "stndt_lite_diverse_screen_m_training_dynamics"
$OutDir = Join-Path $RepoRoot "results\benchmark_runs\$RunName"
$Python = Join-Path $VenvRoot "Scripts\python.exe"
$ResolvedConfig = Join-Path $RepoRoot $ConfigPath
$DefaultDataRoot = Join-Path $RepoRoot "data\raw"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing Python entrypoint: $Python"
}
if (-not (Test-Path -LiteralPath $ResolvedConfig)) {
    throw "Missing config: $ResolvedConfig"
}
if (-not $env:NLB_DATA_DIR -and (Test-Path -LiteralPath $DefaultDataRoot)) {
    $env:NLB_DATA_DIR = $DefaultDataRoot
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$Stamp = Get-Date -Format "yyyyMMddTHHmmss"
$Stdout = Join-Path $OutDir "training_dynamics_$Stamp.out.log"
$Stderr = Join-Path $OutDir "training_dynamics_$Stamp.err.log"
$Status = Join-Path $OutDir "training_dynamics_status_$Stamp.txt"
$PidPath = Join-Path $OutDir "training_dynamics.pid"

$metadata = @(
    "repo_root=$RepoRoot",
    "config=$ConfigPath",
    "entrypoint=$Python",
    "nlb_data_dir=$env:NLB_DATA_DIR",
    "timeout_seconds=$TimeoutSeconds",
    "started_local=$(Get-Date -Format o)",
    "stdout=$Stdout",
    "stderr=$Stderr"
)
$metadata | Set-Content -LiteralPath $Status -Encoding UTF8

if ($DryRun) {
    "dry_run=true" | Add-Content -LiteralPath $Status -Encoding UTF8
    Write-Output "Dry run OK. Would execute: $Python -m nlb_project.cli.run_ensemble_screen --config $ConfigPath --log-level INFO"
    Write-Output "Status: $Status"
    exit 0
}

$proc = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "nlb_project.cli.run_ensemble_screen", "--config", $ConfigPath, "--log-level", "INFO") `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru `
    -WindowStyle Hidden

$proc.Id | Set-Content -LiteralPath $PidPath -Encoding ASCII
"pid=$($proc.Id)" | Add-Content -LiteralPath $Status -Encoding UTF8

$finished = $proc.WaitForExit([Math]::Max(1, $TimeoutSeconds) * 1000)
if (-not $finished) {
    "timed_out_local=$(Get-Date -Format o)" | Add-Content -LiteralPath $Status -Encoding UTF8
    taskkill /PID $proc.Id /T /F | Add-Content -LiteralPath $Status -Encoding UTF8
    Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
    exit 124
}

"finished_local=$(Get-Date -Format o)" | Add-Content -LiteralPath $Status -Encoding UTF8
"exit_code=$($proc.ExitCode)" | Add-Content -LiteralPath $Status -Encoding UTF8
Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
exit $proc.ExitCode
