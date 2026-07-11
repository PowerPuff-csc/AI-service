@echo off
cd /d %~dp0
if not exist venv\Scripts\activate (
  echo Creating Python venv...
  py -3.11 -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
set FLASK_DEBUG=0
echo Starting AI service at http://127.0.0.1:5001
python app.py
