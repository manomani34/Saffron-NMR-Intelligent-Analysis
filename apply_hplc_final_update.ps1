$ProjectRoot = (Get-Location).Path
$ReleaseRoot = Join-Path $ProjectRoot "hplc_release\hplc"

if (!(Test-Path $ReleaseRoot)) {
    throw "hplc_release\hplc not found. Extract the ZIP first."
}

$Targets = @(
    "hplc_dashboard.py",
    "main.py",
    "src\modeling.py",
    "pages\01_origin.py",
    "pages\04_research.py",
    "pages\05_modality_comparison.py"
)

foreach ($rel in $Targets) {
    $src = Join-Path $ReleaseRoot $rel
    $dst = Join-Path $ProjectRoot "hplc\$rel"

    if (!(Test-Path $src)) { throw "Missing release file: $src" }

    $backup = "$dst.before_hplc_final_update"
    if (Test-Path $dst) {
        if (!(Test-Path $backup)) {
            Copy-Item $dst $backup
        }
    }

    Copy-Item $src $dst -Force
    Write-Host "UPDATED: $rel"
}

Write-Host ""
Write-Host "HPLC final update applied."
Write-Host "Backups use: .before_hplc_final_update"
Write-Host ""
Write-Host "Next:"
Write-Host "  python -m hplc.main"
Write-Host "  streamlit run hplc/hplc_dashboard.py"
