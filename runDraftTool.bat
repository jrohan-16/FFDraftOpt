@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Use the folder containing this .bat as working directory
pushd "%~dp0"

REM Optional: make console Unicode-safe (for λ, etc.)
chcp 65001 >nul

REM Prefer a local virtual environment if present, else fall back to Python 3.12, else system python
set "PY_EXE="
if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if not defined PY_EXE if exist "C:\Python312\python.exe" set "PY_EXE=C:\Python312\python.exe"
if not defined PY_EXE set "PY_EXE=python"

echo Using Python: %PY_EXE%
echo.

REM Ensure dependencies. (No-op if already installed)
"%PY_EXE%" -m pip install --quiet --upgrade pip
"%PY_EXE%" -m pip install --quiet streamlit pandas numpy

REM Run the app
"%PY_EXE%" -m streamlit run fantasy_draft_app_refactored.py

popd
pause
