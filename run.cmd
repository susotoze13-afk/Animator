@echo off
REM Double-click launcher for run.ps1. From a terminal you can also call run.ps1 directly.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 pause
