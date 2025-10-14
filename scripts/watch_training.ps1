param(
    [switch]$All,
    [int]$Tail = 50
)

# Find the latest training stdout log
if (-not (Test-Path logs)) {
    Write-Host "No 'logs' directory found in the current folder." -ForegroundColor Yellow
    exit 1
}

$latest = Get-ChildItem -Path "logs" -Filter "train_*.out" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    Write-Host "No training logs found in 'logs'." -ForegroundColor Yellow
    exit 1
}

$log = $latest.FullName
Write-Host "Tailing: $log" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

if ($All) {
    Get-Content -Tail $Tail -Wait $log
} else {
    # Filter to the most useful lines; show last $Tail lines and keep streaming
    Get-Content -Tail $Tail -Wait $log |
        Select-String -Pattern 'Epoch|Step|Validation Results|Causal|Spurious|Checkpoint|Memory|Loss|TRAINING STARTED|STARTING TRAINING|Saving final model|TRAINING COMPLETE'
}

