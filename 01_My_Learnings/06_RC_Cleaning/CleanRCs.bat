@echo off
SET "PY_URL=https://www.python.org"
SET "INSTALLER=%TEMP%\python-installer.exe"
SET "TARGET_DIR=C:\Python312"

:: 1. Check if Python is installed
where python >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo Python is already installed.
    goto end
)

:: 2. Download Installer
echo Python not found. Downloading...
powershell -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%INSTALLER%'"

:: 3. Install with Environment Variables
:: PrependPath=1 adds Python to System PATH permanently
echo Installing and setting environment variables...
start /wait "" "%INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 TargetDir=%TARGET_DIR%

:: 4. Update Current Session Path (Immediate Effect)
:: This allows the script to use 'python' without restarting the CMD window
SET "PATH=%TARGET_DIR%;%TARGET_DIR%\Scripts;%PATH%"

:: 5. Cleanup
del "%INSTALLER%"
echo Python installed and PATH updated.

:end
python --version


cd "00_py_script"
pip install -r requirements.txt
python "01_RC_Cleaning_Script.py"
cd ..
