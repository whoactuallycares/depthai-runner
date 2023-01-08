# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=["/home/tommy/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms/../../../../opencv_contrib_python.libs"],
    binaries=[],
    datas=[("models/out/*", "models/out")],
    hiddenimports=[
        "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
        "sklearn.metrics._pairwise_distances_reduction._middle_term_computer"
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
