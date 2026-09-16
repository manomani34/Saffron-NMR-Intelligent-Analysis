@echo off

echo ==========================================
echo Saffron NMR Dashboard - Build
echo ==========================================

cd /d "%~dp0"

echo.
echo Cleaning previous builds...

rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del /q SaffronNMRDashboard.spec 2>nul

echo.
echo Building EXE...

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --name SaffronNMRDashboard ^
    --collect-all streamlit ^
    --add-data "dashboard.py;." ^
    --add-data "README.md;." ^
    --add-data "run_output.txt;." ^
    --add-data "docs;docs" ^
    --add-data "data;data" ^
    --add-data "reports;reports" ^
    --add-data "outputs;outputs" ^
    --add-data "pca_scores.png;." ^
    --add-data "pca_scree_plot.png;." ^
    --add-data "novelty_detection.png;." ^
    --add-data "shap_summary.png;." ^
    --add-data "robust_evaluation_summary.csv;." ^
    --add-data "robust_evaluation_results.csv;." ^
    --add-data "novelty_detection_results.csv;." ^
    --add-data "sample_predictions.csv;." ^
    --add-data "decision_engine_results.csv;." ^
    --add-data "shap_feature_importance.csv;." ^
    --add-data "final_report.csv;." ^
    launcher.py

if errorlevel 1 (
    echo.
    echo ==========================================
    echo BUILD FAILED
    echo ==========================================
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo BUILD SUCCESSFUL
echo ==========================================
echo.

echo EXE:
echo dist\SaffronNMRDashboard\SaffronNMRDashboard.exe

echo.
pause