HPLC Dashboard Final Polish - v2

The previous PowerShell patch had an encoding problem on Windows PowerShell.
Use this Python patch instead; it avoids ExecutionPolicy and UTF-8 parsing issues.

From the project root:

    python .\hplc_release\apply_hplc_dashboard_polish.py

Then restart Streamlit:

    python -m streamlit run hplc\hplc_dashboard.py

Only these HPLC dashboard files are touched:
- hplc\hplc_dashboard.py
- hplc\pages\01_origin.py
- hplc\pages\04_research.py
- hplc\pages\05_modality_comparison.py

Backups are created once with suffix:
.before_hplc_polish
