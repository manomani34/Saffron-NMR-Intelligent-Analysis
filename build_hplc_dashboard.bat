@echo off

echo ==========================================
echo Saffron HPLC Dashboard - Build
echo ==========================================

cd /d "%~dp0"

echo.
echo Cleaning previous HPLC builds...

rmdir /s /q build_hplc 2>nul
rmdir /s /q dist\SaffronHPLCDashboard 2>nul
del /q SaffronHPLCDashboard.spec 2>nul

echo.
echo Building HPLC EXE...

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --name SaffronHPLCDashboard ^
    --collect-all streamlit ^
    --collect-all plotly ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all pandas ^
    --collect-all openpyxl ^
    --collect-submodules hplc ^
    --add-data "hplc;hplc" ^
    --add-data "components;components" ^
    hplc_launcher.py

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
echo dist\SaffronHPLCDashboard\SaffronHPLCDashboard.exe

echo.
pause