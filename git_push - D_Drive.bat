@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

:: ---------------------------------------------------------------
::  MULTI-REPO GIT SYNC SCRIPT
::  Add your repository paths below (one per line, no trailing \)
:: ---------------------------------------------------------------

SET BRANCH=main
SET PASS=0
SET FAIL=0
SET SKIPPED=0

:: --- ADD YOUR REPO PATHS HERE -----------------------------------
SET REPOS[0]="D:\05 GIT\01_masterRepo"
SET REPOS[1]="D:\05 GIT\14_Proj_ML_99Acres"
SET REPOS[2]="D:\05 GIT\15_AI_Travel_Agent"
SET REPOS[3]="D:\05 GIT\13_Misc_projects"
SET REPOS[4]="D:\05 GIT\12_GenAI_projects"
SET REPOS[5]="D:\05 GIT\11_NLP_projects"
SET REPOS[6]="D:\05 GIT\10_DL_projects"
SET REPOS[7]="D:\05 GIT\09_ML_projects"
SET REPOS[8]="D:\05 GIT\08_WebScraping_projects"
SET REPOS[9]="D:\05 GIT\07_EDA_projects"
SET REPOS[10]="D:\05 GIT\06_Statistical_Analysis_projects"
SET REPOS[11]="D:\05 GIT\05_Python_projects"
SET REPOS[12]="D:\05 GIT\04_PowerBI_projects"
SET REPOS[13]="D:\05 GIT\03_SQL_projects"
SET REPOS[13]="D:\05 GIT\02_Excel_projects"
:: Add more as needed:
:: SET REPOS[3]=C:\path\to\repo4
:: ----------------------------------------------------------------

:: Count repos
SET REPO_COUNT=0
:COUNT_LOOP
    IF DEFINED REPOS[%REPO_COUNT%] (
        SET /A REPO_COUNT+=1
        GOTO COUNT_LOOP
    )

:: ECHO.
:: ECHO +------------------------------------------------------+
:: ECHO ¦           MULTI-REPO GIT SYNC STARTED               ¦
:: ECHO ¦   Total Repositories Found: %REPO_COUNT%                        ¦
:: ECHO +------------------------------------------------------+
:: ECHO.

:: Loop through each repo
SET INDEX=0
:REPO_LOOP
	SET COMMIT_MSG=Auto-Push:__%DATE%__%TIME%
    IF NOT DEFINED REPOS[%INDEX%] GOTO SUMMARY

    SET CURRENT_REPO=!REPOS[%INDEX%]!
    SET /A INDEX+=1

    ECHO +------------------------------------------------------
    ECHO ¦ [REPO %INDEX%] !CURRENT_REPO!
    ECHO +------------------------------------------------------

    :: Check if folder exists
    IF NOT EXIST "!CURRENT_REPO!" (
        ECHO   [ERROR] Path does not exist: !CURRENT_REPO!
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    :: Navigate to repo
    cd /d "!CURRENT_REPO!"
    IF ERRORLEVEL 1 (
        ECHO   [ERROR] Could not navigate to: !CURRENT_REPO!
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    :: Validate it's a git repo
    git rev-parse --is-inside-work-tree >NUL 2>&1
    IF ERRORLEVEL 1 (
        ECHO   [ERROR] Not a git repository: !CURRENT_REPO!
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    :: -- STEP 1: Pull ------------------------------------------
    ECHO   ^> Pulling from origin/%BRANCH%...
    git pull origin %BRANCH% 2>&1
    IF ERRORLEVEL 1 (
        ECHO   [ERROR] git pull failed. Skipping commit/push for this repo.
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    :: -- STEP 2: Status ----------------------------------------
    ECHO.
    ECHO   ^> Checking status...
    git status

    :: Detect changes using --porcelain
    git status --porcelain > "%TEMP%\git_status_%INDEX%.txt" 2>&1
    FOR /F %%A IN ("%TEMP%\git_status_%INDEX%.txt") DO SET FILE_SIZE=%%~zA

    IF "!FILE_SIZE!"=="0" (
        ECHO   [INFO] No changes. Nothing to commit.
        SET /A SKIPPED+=1
        ECHO.
        GOTO REPO_LOOP
    )

    :: -- STEP 3: Add, Commit, Push -----------------------------
    ECHO.
    ECHO   ^> Changes detected! Staging all files...
    git add .

    ECHO   ^> Committing...
    git commit -m "%COMMIT_MSG%"
    IF ERRORLEVEL 1 (
        ECHO   [ERROR] git commit failed.
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    ECHO   ^> Pushing to origin/%BRANCH%...
    git push origin %BRANCH% 2>&1
    IF ERRORLEVEL 1 (
        ECHO   [ERROR] git push failed. Check credentials/remote.
        SET /A FAIL+=1
        ECHO.
        GOTO REPO_LOOP
    )

    ECHO   [SUCCESS] Synced successfully!
    SET /A PASS+=1
    ECHO.
    GOTO REPO_LOOP

:: -- SUMMARY ---------------------------------------------------
:SUMMARY
ECHO +------------------------------------------------------+
ECHO ¦                    SYNC SUMMARY                     ¦
ECHO ¦------------------------------------------------------¦
ECHO ¦  ? Pushed Successfully : !PASS!                              ¦
ECHO ¦  - No Changes (Skipped): !SKIPPED!                              ¦
ECHO ¦  ? Failed              : !FAIL!                              ¦
ECHO +------------------------------------------------------+
:: ECHO.

PAUSE
ENDLOCAL