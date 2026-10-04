from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "hplc_dashboard.py": ROOT / "hplc" / "hplc_dashboard.py",
    "01_origin.py": ROOT / "hplc" / "pages" / "01_origin.py",
    "04_research.py": ROOT / "hplc" / "pages" / "04_research.py",
    "05_modality_comparison.py": ROOT / "hplc" / "pages" / "05_modality_comparison.py",
}


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".before_hplc_polish")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def replace_all(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8-sig")
    original = text

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)

    if text != original:
        backup_once(path)
        path.write_text(text, encoding="utf-8")
        print(f"[FIXED] {path}")
    else:
        print(f"[OK/SKIP] {path}")


# Dashboard home: clarify what 40 means.
replace_all(TARGETS["hplc_dashboard.py"], [
    (
        'st.metric("مشاهدات مستقل", len(independent))',
        'st.metric("رکوردهای یکتا پس از حذف تکرار", len(independent))',
    ),
    (
        'Therefore in analysis **40 independent observations** are used.',
        'Therefore in statistical analysis **40 unique records after duplicate handling** are used.',
    ),
    (
        'با 40 مشاهده مستقل، 11 گروه و چند کلاس 2 نمونه‌ای، ',
        'با 40 رکورد یکتا پس از حذف تکرار، 11 گروه و چند کلاس 2 نمونه‌ای، ',
    ),
])

# Origin: clean RTL/LTR validation note.
replace_all(TARGETS["01_origin.py"], [
    (
        'Validation اکنون provenance-aware است: نمونه‌های 39 و 40 به دلیل رابطه 308 nm در یک Fold قرار می‌گیرند. Balanced Accuracy نهایی از OOF کامل هر Repeat محاسبه شده است.',
        'اعتبارسنجی فعلی provenance-aware است: نمونه‌های 39 و 40 به‌دلیل رابطه مشخص‌شده در 308 nm در یک Fold نگه داشته می‌شوند. Balanced Accuracy به‌صورت OOF کامل هر Repeat محاسبه شده است.',
    ),
])

# Research: same terminology.
replace_all(TARGETS["04_research.py"], [
    (
        '["مشاهدات مستقل", len(independent)],',
        '["رکوردهای یکتا پس از حذف تکرار", len(independent)],',
    ),
    (
        '**Sample 32 / 308 nm:** تنها یک نقطه نامعتبر دارد و حذف کل نمونه لازم نیست؛',
        '**Sample 32 / 308 nm:** تنها یک نقطه نامعتبر دارد و حذف کل رکورد لازم نیست؛',
    ),
])

# Modality comparison: avoid implying Combined is best from Accuracy alone.
replace_all(TARGETS["05_modality_comparison.py"], [
    (
        'st.warning(\n    "این صفحه رتبه‌بندی عملیاتی ارائه نمی‌کند. هدف فقط نشان دادن تفاوت توصیفی modalityها تحت یک validation مشترک است."\n)',
        'st.warning(\n    "این صفحه رتبه‌بندی عملیاتی ارائه نمی‌کند. هدف فقط نشان دادن تفاوت توصیفی modalityها تحت یک validation مشترک است."\n)\n\nst.info(\n    "ترکیب سه طول موج (Combined) الزاماً به بهبود Balanced Accuracy منجر نشده است؛ بنابراین Accuracy بالاتر آن به‌تنهایی به معنی عملکرد جغرافیایی بهتر نیست."\n)',
    ),
])

print()
print("=== PYTHON SYNTAX CHECK ===")

for path in TARGETS.values():
    source = path.read_text(encoding="utf-8-sig")
    compile(source, str(path), "exec")
    print(f"[PASS] {path.name}")

print()
print("HPLC dashboard polish completed successfully.")
print("Backups use suffix: .before_hplc_polish")
