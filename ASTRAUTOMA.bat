@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title ASTRAUTOMA
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem ASTRAUTOMA launcher. ASCII only + CRLF: cmd.exe breaks on anything else.

rem 1. Find the real python.exe here (Windows Terminal starts with its own PATH)
set "PYEXE="
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE (
    echo Python 3.10+ not found. Install it from python.org.
    pause
    exit /b 1
)

rem 2. Dependencies
"%PYEXE%" -c "import krpc, PIL" >nul 2>&1 || "%PYEXE%" -m pip install -q -r requirements.txt

rem 3. Full screen in Windows Terminal if available, otherwise this window
where wt >nul 2>&1
if not errorlevel 1 (
    start "" wt --fullscreen --title ASTRAUTOMA -d "%~dp0." "%PYEXE%" -X utf8 "%~dp0astrautoma.py" %*
    exit /b 0
)
mode con: cols=160 lines=50 >nul 2>&1
"%PYEXE%" -X utf8 astrautoma.py %*
if errorlevel 1 pause
