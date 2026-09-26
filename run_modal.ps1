# Modal Execution Helper for Amazon ML Challenge 2026
# Run from Amazon-ML-challenge/ directory or project root
param(
    [switch]$Sync,
    [switch]$Quick,
    [switch]$Baseline,
    [string]$Embedder = "",     # bgem3, e5small, e5base, minilm, none
    [switch]$Loco,
    [string]$Gpu = "A100",      # A100 or A10G
    [switch]$Status,
    [switch]$Setup
)

$ErrorActionPreference = "Stop"

# Use the Python 3.12 executable where modal is installed
$MODAL_EXE = "C:\Users\navya\AppData\Local\Programs\Python\Python312\Scripts\modal.exe"
if (-not (Test-Path $MODAL_EXE)) {
    $MODAL_EXE = "modal"
}

if ($Setup) {
    Write-Host "=== Setting up Modal Authentication ===" -ForegroundColor Cyan
    & $MODAL_EXE setup
    exit
}

if ($Status) {
    Write-Host "=== Modal Account & Volume Status ===" -ForegroundColor Cyan
    Write-Host "Checking current profile:" -ForegroundColor Yellow
    & $MODAL_EXE profile current
    Write-Host "`nChecking volumes:" -ForegroundColor Yellow
    & $MODAL_EXE volume list
    exit
}

if ($Sync) {
    Write-Host "=== Uploading Dataset to Modal Cloud Volume 'amazon-ml-data' ===" -ForegroundColor Cyan
    & $MODAL_EXE run modal_runner.py::sync_dataset
    exit
}

# Pipeline Execution
Write-Host "=== Launching Pipeline on Modal Cloud ($Gpu) ===" -ForegroundColor Cyan
$cmdArgs = @("run", "modal_runner.py", "--gpu", $Gpu)

if ($Quick) {
    $cmdArgs += "--quick"
}

if ($Baseline) {
    $cmdArgs += "--embedder"
    $cmdArgs += "none"
    $cmdArgs += "--loco"
} elseif ($Embedder -ne "") {
    $cmdArgs += "--embedder"
    $cmdArgs += $Embedder
    $cmdArgs += "--loco"
} else {
    # Default if no embedder or baseline specified
    if (-not $Quick) {
        $cmdArgs += "--loco"
    }
}

Write-Host "Running: $MODAL_EXE $($cmdArgs -join ' ')" -ForegroundColor DarkGray
& $MODAL_EXE @cmdArgs
