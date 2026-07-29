@echo off
title NL2SQL Agent

echo ============================================
echo   NL2SQL Agent
echo ============================================
echo.

:: 清端口
echo [1/2] 清理端口...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860"') do taskkill /F /PID %%a >nul 2>&1

:: 启动 Agent
echo [2/2] 启动 Agent + ngrok...
start "Agent" cmd /k "cd /d %~dp0 && python app.py"
timeout /t 8 /nobreak >nul
:: 读取固定域名（如果配了）
set NG_DOMAIN=
for /f "tokens=2 delims=: " %%a in ('findstr /r "^domain:" "%~dp0ngrok.yml" 2^>nul') do set NG_DOMAIN=%%a
if defined NG_DOMAIN (
    echo       固定域名: %NG_DOMAIN%
    start "ngrok" cmd /k "cd /d %~dp0 && ngrok http 7860 --config=ngrok.yml --domain=%NG_DOMAIN%"
) else (
    start "ngrok" cmd /k "cd /d %~dp0 && ngrok http 7860 --config=ngrok.yml"
)

echo.
echo ============================================
echo   Agent:  http://localhost:7860
echo   ngrok:  查看弹出窗口里的 Forwarding 链接
echo           发给朋友即可
echo ============================================

timeout /t 3 /nobreak >nul
