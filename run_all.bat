@echo off
REM Run training for every config. Activate venv first if you use one.
cd /d "%~dp0"

for %%f in (configs\*.yaml) do (
    echo.
    echo ========== Running: %%f ==========
    python train.py --config %%f
    if errorlevel 1 (
        echo FAILED: %%f
        pause
        exit /b 1
    )
)

echo.
echo All 9 experiments finished.
pause
