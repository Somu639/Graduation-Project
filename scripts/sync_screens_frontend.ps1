# Sync Stitch export from screens/ into frontend/web/ (run after updating the zip export).
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "screens\stitch_ai_review_discovery_dashboard"
$dest = Join-Path $root "frontend\web"

foreach ($name in @("code.html", "DESIGN.md")) {
    Copy-Item (Join-Path $src $name) (Join-Path $dest $name) -Force
}
Write-Host "Synced $src -> $dest"
