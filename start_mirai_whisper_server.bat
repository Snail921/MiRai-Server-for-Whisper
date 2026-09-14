@echo off
setlocal EnableExtensions
title MiRAI Whisper OpenAI-Compatible Server
set "APP_ROOT=%~dp0"
set "SERVER_PYTHON="

rem Prefer an explicit override, then common virtual-environment folders.
if defined MIRAI_WHISPER_PYTHON call :select_python "%MIRAI_WHISPER_PYTHON%"
if not defined SERVER_PYTHON if defined WHISPER_ASR_PYTHON call :select_python "%WHISPER_ASR_PYTHON%"
if not defined SERVER_PYTHON call :select_python "%APP_ROOT%env\Scripts\python.exe"
if not defined SERVER_PYTHON call :select_python "%APP_ROOT%.venv\Scripts\python.exe"
if not defined SERVER_PYTHON call :select_python "%APP_ROOT%venv\Scripts\python.exe"
if not defined SERVER_PYTHON call :select_python "%APP_ROOT%Scripts\python.exe"

rem Keep existing installations that use set_runtime_env.bat working.
if not defined SERVER_PYTHON if exist "%APP_ROOT%set_runtime_env.bat" call :select_legacy_runtime

rem Finally try Python executables already available on PATH.
if not defined SERVER_PYTHON (
  for /f "delims=" %%P in ('where python.exe 2^>nul') do call :select_python "%%P"
)

if not defined SERVER_PYTHON (
  echo [ERROR] A Python environment containing openai-whisper was not found.
  echo.
  echo Place these files beside env, .venv, or venv, or set:
  echo   MIRAI_WHISPER_PYTHON=C:\path\to\python.exe
  goto :failed
)

if not exist "%APP_ROOT%mirai_whisper_server.py" (
  echo [ERROR] Missing: "%APP_ROOT%mirai_whisper_server.py"
  goto :failed
)

rem A Whisper-only environment may not have the HTTP server dependencies yet.
"%SERVER_PYTHON%" -c "import fastapi, uvicorn, multipart, webrtcvad" >nul 2>&1
if errorlevel 1 (
  if not exist "%APP_ROOT%requirements-server.txt" (
    echo [ERROR] Server dependencies are missing, and requirements-server.txt was not found.
    goto :failed
  )
  echo [SETUP] Installing MiRAI Whisper server dependencies...
  "%SERVER_PYTHON%" -m pip install -r "%APP_ROOT%requirements-server.txt"
  if errorlevel 1 (
    echo [ERROR] Failed to install the server dependencies.
    goto :failed
  )
  "%SERVER_PYTHON%" -c "import fastapi, uvicorn, multipart, webrtcvad" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] The server dependencies are still unavailable after installation.
    goto :failed
  )
)

echo Python: "%SERVER_PYTHON%"
echo Listening on all IPv4 interfaces at port 8000.
echo Use this PC's LAN IPv4 address from MiRAI Head.
echo.
"%SERVER_PYTHON%" "%APP_ROOT%mirai_whisper_server.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Server stopped. Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%

:select_python
if defined SERVER_PYTHON exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import whisper, torch" >nul 2>&1
if errorlevel 1 exit /b 0
set "SERVER_PYTHON=%~1"
exit /b 0

:select_legacy_runtime
call "%APP_ROOT%set_runtime_env.bat"
if defined WHISPER_ASR_PYTHON call :select_python "%WHISPER_ASR_PYTHON%"
exit /b 0

:failed
echo.
echo MiRAI Whisper server was not started.
pause
exit /b 1
