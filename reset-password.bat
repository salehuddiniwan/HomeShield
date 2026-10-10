@echo off
rem Double-click on the computer that runs HomeShield when no admin can sign in.
rem It gives one account a temporary password; nothing else changes.
setlocal
cd /d "%~dp0"

rem Find HomeShield's Python: HOMESHIELD_PYTHON if set, then the conda
rem environments named in the README, then a .venv folder, then PATH.
set "PY="
if defined HOMESHIELD_PYTHON set "PY=%HOMESHIELD_PYTHON%"
for %%E in (homeshield venv) do (
  for %%B in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if not defined PY if exist "%%~B\envs\%%E\python.exe" set "PY=%%~B\envs\%%E\python.exe"
  )
)
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo HomeShield password reset
echo.
set "HS_USER="
set /p "HS_USER=Username to reset (press Enter for admin): "
if not defined HS_USER set "HS_USER=admin"
echo.
"%PY%" run_homeshield.py --reset-password "%HS_USER%" %*
if errorlevel 1 (
  echo.
  echo That didn't work. Check the username, or see "Forgot the admin password" in README.md.
)
echo.
pause
