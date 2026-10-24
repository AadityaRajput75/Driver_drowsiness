@echo off
setlocal
if exist "%~dp0.venv\Scripts\python.exe" goto project_python
if exist "%LOCALAPPDATA%\Python\bin\python.exe" goto local_python
if exist "%USERPROFILE%\AppData\Local\Python\bin\python.exe" goto user_python
where py >nul 2>nul
if not errorlevel 1 goto python_launcher
python "%~dp0main.py" %*
exit /b %errorlevel%

:project_python
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
exit /b %errorlevel%

:local_python
"%LOCALAPPDATA%\Python\bin\python.exe" "%~dp0main.py" %*
exit /b %errorlevel%

:user_python
"%USERPROFILE%\AppData\Local\Python\bin\python.exe" "%~dp0main.py" %*
exit /b %errorlevel%

:python_launcher
py -3 "%~dp0main.py" %*
exit /b %errorlevel%
