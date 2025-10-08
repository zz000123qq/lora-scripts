@echo off
chcp 65001

REM ----------------------------------------------------------------
REM SD-Trainer 启动脚本 - 【局域网公开访问模式】
REM ----------------------------------------------------------------

REM 提示：如果你的 PyCharm 终端里的 Python 3.13 运行良好，通常不需要手动指定路径。
set PYTHON_EXEC=python.exe
set HOST_IP=0.0.0.0
set PORT=28000

REM --- 1. 检查 Python 可执行文件 ---
where %PYTHON_EXEC% >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误：找不到 Python.exe。
    echo 请检查系统环境变量 PATH。
    pause
    exit /b 1
)

REM --- 2. 启动 GUI 主程序 ---
echo ---------------------------------------------------
echo 正在启动 SD-Trainer GUI...
echo 服务器将监听所有网络接口 (0.0.0.0)。
echo 访问地址： 请在浏览器中输入本机的局域网IP地址:%PORT%
echo ---------------------------------------------------

REM 执行 gui.py 并使用 --host 0.0.0.0 参数
"%PYTHON_EXEC%" gui.py --host %HOST_IP% --port %PORT%

REM --- 3. 保持窗口开启 (如果程序意外退出) ---
echo.
echo SD-Trainer 已停止运行。
pause