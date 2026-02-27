@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  Python Auto-Installer Script
::  Author : Gaurav Singh
::  Desc   : Checks if Python is installed; installs if not.
:: ============================================================

echo ============================================
echo   Python Installation Checker ^& Installer
echo ============================================
echo.

:: ------------------------------------
:: STEP 1: Check if Python is installed
:: ------------------------------------
where python >nul 2>&1
if %errorlevel% == 0 (
    for /f "delims=" %%V in ('python --version 2^>^&1') do set PYVER=%%V
    echo [INFO] Python is already installed: !PYVER!
    echo [INFO] Installation path:
    where python
    goto :SetEnvVars
) else (
    echo [WARN] Python is NOT installed on this system.
    echo [INFO] Proceeding with automatic installation...
    goto :InstallPython
)

:: ------------------------------------
:: STEP 2: Download and Install Python
:: ------------------------------------
:InstallPython

:: Set desired Python version and download URL
set PYTHON_VERSION=3.12.9
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
set INSTALLER_PATH=%TEMP%\python-%PYTHON_VERSION%-amd64.exe
set INSTALL_DIR=C:\Python312

echo [INFO] Downloading Python %PYTHON_VERSION% from official source...
echo [INFO] URL: %PYTHON_URL%
echo.

:: Download using PowerShell (built-in on all modern Windows)
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%INSTALLER_PATH%' -UseBasicParsing"

if not exist "%INSTALLER_PATH%" (
    echo [ERROR] Download failed! Check your internet connection.
    pause
    exit /b 1
)

echo [INFO] Download complete. Starting silent installation...
echo.

:: Silent install with all important flags
:: InstallAllUsers=1    -> Install for all users (requires admin)
:: PrependPath=1        -> Add Python to PATH automatically
:: Include_pip=1        -> Install pip
:: Include_launcher=1  -> Install Python launcher (py command)
:: DefaultCustomInstall=1 + DefaultPath -> Custom install directory
"%INSTALLER_PATH%" /quiet ^
    InstallAllUsers=1 ^
    PrependPath=1 ^
    Include_pip=1 ^
    Include_launcher=1 ^
    DefaultCustomInstall=1 ^
    DefaultPath="%INSTALL_DIR%"

if %errorlevel% neq 0 (
    echo [ERROR] Python installation failed with error code: %errorlevel%
    del /f /q "%INSTALLER_PATH%" >nul 2>&1
    pause
    exit /b 1
)

echo [INFO] Python %PYTHON_VERSION% installed successfully!
echo.

:: Cleanup installer
del /f /q "%INSTALLER_PATH%" >nul 2>&1
echo [INFO] Installer cleaned up from TEMP folder.

:: ------------------------------------
:: STEP 3: Set Environment Variables
:: ------------------------------------
:SetEnvVars

echo.
echo [INFO] Configuring Environment Variables...

:: Determine install path (use default if newly installed, or detect existing)
where python >nul 2>&1
if %errorlevel% == 0 (
    for /f "delims=" %%P in ('where python') do (
        set PYTHON_EXE_PATH=%%P
        goto :GotPath
    )
)
set PYTHON_EXE_PATH=%INSTALL_DIR%\python.exe

:GotPath
:: Extract directory from full path
for %%F in ("%PYTHON_EXE_PATH%") do set PYTHON_BIN_DIR=%%~dpF
:: Remove trailing backslash
if "%PYTHON_BIN_DIR:~-1%"=="\" set PYTHON_BIN_DIR=%PYTHON_BIN_DIR:~0,-1%
set PYTHON_SCRIPTS_DIR=%PYTHON_BIN_DIR%\Scripts

echo [INFO] Python binary directory : %PYTHON_BIN_DIR%
echo [INFO] Python Scripts directory: %PYTHON_SCRIPTS_DIR%

:: Set PYTHONPATH system environment variable (Machine-level)
setx PYTHONPATH "%PYTHON_BIN_DIR%" /M
if %errorlevel% == 0 (
    echo [OK]   PYTHONPATH set to: %PYTHON_BIN_DIR%
) else (
    echo [WARN] Failed to set PYTHONPATH. Try running as Administrator.
)

:: Add Python and Scripts to System PATH (Machine-level)
for /f "tokens=2*" %%A in (
    'reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul'
) do set "CURRENT_PATH=%%B"

:: Check if already in PATH to avoid duplicates
echo !CURRENT_PATH! | find /i "%PYTHON_BIN_DIR%" >nul
if %errorlevel% neq 0 (
    setx PATH "!CURRENT_PATH!;%PYTHON_BIN_DIR%;%PYTHON_SCRIPTS_DIR%" /M
    echo [OK]   Added Python and Scripts to System PATH.
) else (
    echo [INFO] Python directories already exist in System PATH. No changes needed.
)

:: ------------------------------------
:: STEP 4: Verify Installation
:: ------------------------------------
echo.
echo [INFO] Verifying Python installation...
echo [INFO] (Note: Open a new CMD window if 'python' is not recognized below)
echo.

:: Refresh PATH in current session
set "PATH=%PYTHON_BIN_DIR%;%PYTHON_SCRIPTS_DIR%;%PATH%"

python --version
if %errorlevel% == 0 (
    echo [OK]   Python is ready to use!
    python -m pip --version
) else (
    echo [WARN] Python command not found in current session.
    echo [INFO] Please close and reopen Command Prompt for PATH changes to take effect.
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.




cd "00_py_script"
pip install -r requirements.txt
python "01_RC_Cleaning_Script.py"
cd ..

endlocal