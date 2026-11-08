@echo off
rem Starts the ALR-UVP6 app in the web browser (double-click this file).
rem Uses the conda environment alr_uvp when Miniconda or Anaconda is installed
rem in one of the usual folders, otherwise the Python found on the PATH.
cd /d "%~dp0"
for %%C in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~C\condabin\conda.bat" (
        call "%%~C\condabin\conda.bat" activate alr_uvp
        goto run
    )
)
:run
python -c "import streamlit" 2>nul || python -m pip install -r requirements.txt
rem Streamlit asks for an e-mail address on its first start: an empty one is saved here, so it does not wait
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)
python -m streamlit run app.py --browser.gatherUsageStats false
pause
