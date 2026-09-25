@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=psdenv"

rem Create the virtual environment (py launcher first: "python" is often not on PATH on Windows)
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv "%VENV_DIR%" 2>nul || python -m venv "%VENV_DIR%" || goto :error
)

rem Install requirements when requirements.txt changed since the last install
fc /b requirements.txt "%VENV_DIR%\requirements.installed" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
    copy /y requirements.txt "%VENV_DIR%\requirements.installed" >nul
)

"%VENV_DIR%\Scripts\python.exe" main.py
goto :end

:error
echo Setup failed.

:end
pause
