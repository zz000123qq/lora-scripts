@echo off
chcp 65001

REM -----------------------------------------------------------
REM SD-Trainer 启动脚本 (为定制化环境优化)
REM -----------------------------------------------------------

REM --- 1. 设置路径 (如果需要) ---
REM 确保你的系统环境变量 PATH 中包含正确的 Python 3.13 路径。
REM 如果 Python 不在 PATH 中，你需要手动指定路径。
REM 例如： set PYTHON_EXEC="C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"

REM 默认使用系统中的 Python
set PYTHON_EXEC=python.exe

REM --- 2. 检查 Python 可执行文件 ---
where %PYTHON_EXEC% >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误：找不到 Python.exe。
    echo 请检查您的系统环境变量 PATH，或者手动编辑此文件并设置 PYTHON_EXEC 变量。
    pause
    exit /b 1
)

REM --- 3. 启动 GUI 主程序 ---
echo 正在使用 %PYTHON_EXEC% 启动 SD-Trainer GUI...

REM 执行 gui.py
"%PYTHON_EXEC%" gui.py

REM --- 4. 保持窗口开启 (如果程序意外退出) ---
echo.
echo SD-Trainer 已停止运行。
pause