@echo off

REM Adjust if your Conda lives somewhere else
SET "CONDA_ROOT=%USERPROFILE%\anaconda3"
IF NOT EXIST "%CONDA_ROOT%\Scripts\activate.bat" (
    SET "CONDA_ROOT=%USERPROFILE%\Miniconda3"
)

IF NOT EXIST "%CONDA_ROOT%\Scripts\activate.bat" (
    echo ERROR: Cannot locate conda. Edit CONDA_ROOT at top of this script.
    pause
    exit /b 1
)

CALL "%CONDA_ROOT%\Scripts\activate.bat"

REM Create + activate
CALL conda activate rlgym-trueskill

python -m build
pause
