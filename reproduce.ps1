[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CocoPath,

    [string]$OutputDir = "",
    [string]$CheckpointPath = "",
    [switch]$QuickCheck,
    [switch]$SkipInstall,
    [switch]$Visualize,
    [string]$PythonExecutable = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$ReferencePath = Join-Path $RepoRoot "results\reference\official_detr_r50_coco_val2017.json"
$VenvPython = Join-Path $RepoRoot ".venv-repro\Scripts\python.exe"
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "artifacts\runs\windows-eval"
}
if (-not $CheckpointPath) {
    $CheckpointPath = Join-Path $RepoRoot "artifacts\weights\detr-r50-e632da11.pth"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "> $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

Set-Location $RepoRoot

if ($PythonExecutable) {
    $Python = $PythonExecutable
} else {
    $Python = $VenvPython
    if (-not (Test-Path -LiteralPath $Python)) {
        if ($SkipInstall) {
            throw "The local .venv-repro does not exist. Remove -SkipInstall or pass -PythonExecutable."
        }
        $BootstrapPython = (Get-Command python -ErrorAction Stop).Source
        Invoke-Checked $BootstrapPython @("-m", "venv", (Join-Path $RepoRoot ".venv-repro"))
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

if (-not $SkipInstall) {
    Invoke-Checked $Python @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $Python @(
        "-m", "pip", "install",
        "torch==2.6.0", "torchvision==0.21.0",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    )
    Invoke-Checked $Python @("-m", "pip", "install", "-r", (Join-Path $RepoRoot "requirements-repro.txt"))
}

Invoke-Checked $Python @(
    "-c",
    "import sys, torch; print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU available: {torch.cuda.is_available()}')"
)

Invoke-Checked $Python @(
    (Join-Path $RepoRoot "scripts\verify_dataset.py"),
    "--coco-root", $CocoPath,
    "--split", "val",
    "--expected-images", "5000",
    "--check-files"
)

$Reference = Get-Content -LiteralPath $ReferencePath -Raw | ConvertFrom-Json
Invoke-Checked $Python @(
    (Join-Path $RepoRoot "scripts\download_checkpoint.py"),
    "--url", $Reference.checkpoint.url,
    "--sha256", $Reference.checkpoint.sha256,
    "--destination", $CheckpointPath
)

Invoke-Checked $Python @("-m", "unittest", "discover", "-s", (Join-Path $RepoRoot "tests"), "-v")

if ($QuickCheck) {
    Write-Host "Quick check passed. Full COCO evaluation was not started." -ForegroundColor Green
    exit 0
}

$MetricsPath = Join-Path $OutputDir "eval_metrics.json"
if (Test-Path -LiteralPath $MetricsPath) {
    throw "Refusing to overwrite an existing evaluation: $MetricsPath. Choose a new -OutputDir."
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$EvalArguments = @(
    (Join-Path $RepoRoot "main.py"),
    "--batch_size", "2",
    "--num_workers", "2",
    "--eval",
    "--resume", $CheckpointPath,
    "--coco_path", $CocoPath,
    "--output_dir", $OutputDir
)
$RecordedCommand = @($Python) + $EvalArguments
Invoke-Checked $Python (@(
    (Join-Path $RepoRoot "scripts\record_manifest.py"),
    "--output-dir", $OutputDir,
    "--config", $ReferencePath
) + $RecordedCommand)

Invoke-Checked $Python $EvalArguments
Invoke-Checked $Python @(
    (Join-Path $RepoRoot "scripts\verify_metrics.py"),
    "--actual", $MetricsPath,
    "--reference", $ReferencePath
)

if ($Visualize) {
    Invoke-Checked $Python @(
        (Join-Path $RepoRoot "scripts\visualize_predictions.py"),
        "--coco-path", $CocoPath,
        "--checkpoint", $CheckpointPath,
        "--output-dir", (Join-Path $OutputDir "visualizations"),
        "--threshold", "0.70"
    )
}

Write-Host "Evaluation completed and verified: $OutputDir" -ForegroundColor Green
