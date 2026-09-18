# HPLC Saffron Analysis

این بخش یک Pipeline و داشبورد مستقل برای تحلیل HPLC زعفران است و از کد و خروجی‌های NMR استفاده نمی‌کند.

## ساختار

- `main.py` — اجرای کامل Pipeline و تولید خروجی‌ها
- `hplc_dashboard.py` — داشبورد فقط برای نمایش خروجی‌های ذخیره‌شده
- `src/` — بارگذاری داده، Audit، preprocessing و مدل‌سازی
- `data/raw/hplc.xlsx` — فایل خام HPLC
- `data/mapping/sample_mapping.csv` — Mapping مرجع مشتری
- `outputs/` — نتایج محاسبات

## اجرای Pipeline

از ریشه پروژه:

```powershell
python .\hplc\main.py
```

بعد داشبورد:

```powershell
python -m streamlit run .\hplc\hplc_dashboard.py
```

## اصل معماری

محاسبات سنگین در `main.py` انجام می‌شوند و در `outputs/` ذخیره می‌شوند. داشبورد فقط این فایل‌ها را می‌خواند و هنگام باز شدن مدل را دوباره اجرا نمی‌کند.

## داده و Mapping

سه طول موج به‌صورت جداگانه تحلیل می‌شوند:

- 440 nm — Crocin
- 250 nm — Picrocrocin
- 308 nm — Safranal

تحلیل `Combined` با الحاق سه ماتریس پس از preprocessing متناظر انجام می‌شود. Mapping مشتری مرجع نهایی است.

چهار زوج تکراری به یک HPLC measurement مربوط‌اند و در تحلیل مستقل فقط یک مشاهده از هر زوج استفاده می‌شود:

`1/2`, `24/25`, `28/29`, `34/35`

نمونه 13 در Sheet سافرانال با G9 ثبت شده بود، اما Mapping نهایی مشتری G4 است؛ تحلیل از G4 استفاده می‌کند.

## Robust Evaluation

ارزیابی Robust فعلی با:

- `RepeatedStratifiedKFold`
- `2-Fold × 20 Repeats`
- در مجموع `40` split برای هر ترکیب preprocessing/modality
- `PLS-DA` با 4 مؤلفه
- Imputation و StandardScaler داخل هر Fold
\`Raw`, `SNV`, `AsLS + SNV`, و `AsLS + L2` در چهار modality شامل 440، 250، 308 و Combined بررسی می‌شوند.

برای جلوگیری از انتقال اطلاعات آزمون به آموزش، Imputation و StandardScaler فقط روی Training Fold برازش می‌شوند. تبدیل‌های SNV و AsLS به‌صورت deterministic روی هر کروماتوگرام اعمال می‌شوند و از label استفاده نمی‌کنند.

این ارزیابی پژوهشی/اکتشافی است و به دلیل 40 مشاهده مستقل، 11 گروه و کوچک‌ترین کلاس 2 نمونه‌ای، هیچ Winner عملیاتی یا نتیجه قطعی برای استقرار ارائه نمی‌کند.

## نکات داده خام

فایل HPLC فعلی تا 30.0 دقیقه داده دارد، در حالی که در توضیحات مشتری زمان تقریبی Safranal حدود 30.9 دقیقه ذکر شده است. برای 30.9 دقیقه هیچ extrapolation انجام نمی‌شود.
