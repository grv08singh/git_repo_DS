@echo off

cd "00_py_script"
pip install -r requirements.txt
python "01_RC_Cleaning_Script.py"
cd ..
pause
endlocal