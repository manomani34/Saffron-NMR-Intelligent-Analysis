# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('dashboard.py', '.'), ('README.md', '.'), ('run_output.txt', '.'), ('docs', 'docs'), ('data', 'data'), ('reports', 'reports'), ('outputs', 'outputs'), ('pca_scores.png', '.'), ('pca_scree_plot.png', '.'), ('novelty_detection.png', '.'), ('shap_summary.png', '.'), ('robust_evaluation_summary.csv', '.'), ('robust_evaluation_results.csv', '.'), ('novelty_detection_results.csv', '.'), ('sample_predictions.csv', '.'), ('decision_engine_results.csv', '.'), ('shap_feature_importance.csv', '.'), ('final_report.csv', '.')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SaffronNMRDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SaffronNMRDashboard',
)
