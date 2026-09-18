
$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
$Targets = @(
    "$Root\hplc\hplc_dashboard.py",
    "$Root\hplc\pages\01_origin.py",
    "$Root\hplc\pages\04_research.py",
    "$Root\hplc\pages\05_modality_comparison.py"
)

function Backup-Once([string]$Path) {
    $Backup = "$Path.before_hplc_polish"
    if (Test-Path $Path) {
        if (-not (Test-Path $Backup)) { Copy-Item $Path $Backup }
    }
}

function Replace-Exact([string]$Path, [string]$Old, [string]$New) {
    if (-not (Test-Path $Path)) { throw "File not found: $Path" }
    $Text = Get-Content -Raw -Encoding UTF8 $Path
    if ($Text.Contains($Old)) {
        Backup-Once $Path
        $Text = $Text.Replace($Old, $New)
        Set-Content -Path $Path -Value $Text -Encoding UTF8
        Write-Host "[FIXED] $([IO.Path]::GetFileName($Path))"
    } else {
        Write-Host "[OK/SKIP] $([IO.Path]::GetFileName($Path))"
    }
}

# 1) Avoid the ambiguous phrase "independent observations".
Replace-Exact "$Root\hplc\hplc_dashboard.py" `
    'st.metric("مشاهدات مستقل", len(independent))' `
    'st.metric("رکوردهای یکتا پس از حذف تکرار", len(independent))'

Replace-Exact "$Root\hplc\hplc_dashboard.py" `
    'Therefore in analysis **40 independent observations** are used.' `
    'Therefore in statistical analysis **40 unique records after duplicate handling** are used.'

Replace-Exact "$Root\hplc\hplc_dashboard.py" `
    'با 40 مشاهده مستقل، 11 گروه و چند کلاس 2 نمونه‌ای، ' `
    'با 40 رکورد یکتا پس از حذف تکرار، 11 گروه و چند کلاس 2 نمونه‌ای، '

# 2) Origin validation note: more readable RTL/LTR wording.
Replace-Exact "$Root\hplc\pages\01_origin.py" `
    'Validation اکنون provenance-aware است: نمونه‌های 39 و 40 به دلیل رابطه 308 nm در یک Fold قرار می‌گیرند. Balanced Accuracy نهایی از OOF کامل هر Repeat محاسبه شده است.' `
    'اعتبارسنجی فعلی provenance-aware است: نمونه‌های 39 و 40 به‌دلیل رابطه مشخص‌شده در 308 nm در یک Fold نگه داشته می‌شوند. Balanced Accuracy به‌صورت OOF کامل هر Repeat محاسبه شده است.'

# 3) Research page: same precise terminology.
Replace-Exact "$Root\hplc\pages\04_research.py" `
    '["مشاهدات مستقل", len(independent)],' `
    '["رکوردهای یکتا پس از حذف تکرار", len(independent)],'

Replace-Exact "$Root\hplc\pages\04_research.py" `
    '**Sample 32 / 308 nm:** تنها یک نقطه نامعتبر دارد و حذف کل نمونه لازم نیست؛' `
    '**Sample 32 / 308 nm:** تنها یک نقطه نامعتبر دارد و حذف کل رکورد لازم نیست؛'

# 4) Modality comparison: explicitly prevent the wrong "Combined is best" reading.
Replace-Exact "$Root\hplc\pages\05_modality_comparison.py" `
    'st.warning(\n    "این صفحه رتبه‌بندی عملیاتی ارائه نمی‌کند. هدف فقط نشان دادن تفاوت توصیفی modalityها تحت یک validation مشترک است."\n)' `
    'st.warning(\n    "این صفحه رتبه‌بندی عملیاتی ارائه نمی‌کند. هدف فقط نشان دادن تفاوت توصیفی modalityها تحت یک validation مشترک است."\n)\n\nst.info(\n    "ترکیب سه طول موج (Combined) الزاماً به بهبود Balanced Accuracy منجر نشده است؛ بنابراین Accuracy بالاتر آن به‌تنهایی به معنی عملکرد جغرافیایی بهتر نیست."\n)'

Write-Host ""
Write-Host "=== SYNTAX CHECK ==="
$Python = Get-Command python -ErrorAction Stop
python -m py_compile `
    "$Root\hplc\hplc_dashboard.py" `
    "$Root\hplc\pages\01_origin.py" `
    "$Root\hplc\pages\04_research.py" `
    "$Root\hplc\pages\05_modality_comparison.py"

Write-Host ""
Write-Host "HPLC dashboard polish completed successfully."
Write-Host "Backups use suffix: .before_hplc_polish"
