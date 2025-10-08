import argparse
import locale
import os
import platform
import subprocess
import sys
from pathlib import Path

from mikazuki.launch_utils import (base_dir_path, catch_exception, git_tag,
                                   check_port_avaliable, find_avaliable_ports)
from mikazuki.log import log

# --- 参数定义保持不变 ---
parser = argparse.ArgumentParser(description="GUI for stable diffusion training")
# ... (所有 parser.add_argument... 的代码都保持原样)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=28000, help="Port to run the server on")
parser.add_argument("--listen", action="store_true")
parser.add_argument("--skip-prepare-environment", action="store_true")
parser.add_argument("--skip-prepare-onnxruntime", action="store_true")
parser.add_argument("--disable-tensorboard", action="store_true")
parser.add_argument("--disable-tageditor", action="store_true")
parser.add_argument("--disable-auto-mirror", action="store_true")
parser.add_argument("--tensorboard-host", type=str, default="127.0.0.1", help="Port to run the tensorboard")
parser.add_argument("--tensorboard-port", type=int, default=6006, help="Port to run the tensorboard")
parser.add_argument("--localization", type=str)
parser.add_argument("--dev", action="store_true")


@catch_exception
def run_tensorboard(args):
    """启动 TensorBoard 子进程"""
    log.info("Starting tensorboard...")
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", "logs",
        "--host", args.tensorboard_host,
        "--port", str(args.tensorboard_port)
    ]
    subprocess.Popen(cmd)


@catch_exception
def run_tag_editor(args):
    """启动标签编辑器子进程"""
    log.info("Starting tageditor...")
    # 使用 pathlib 构建路径，更清晰健壮
    editor_script = base_dir_path() / "mikazuki" / "dataset-tag-editor" / "scripts" / "launch.py"

    cmd = [
        sys.executable,
        str(editor_script),
        "--port", "28001",
        "--shadow-gradio-output",
        "--root-path", "/proxy/tageditor"
    ]

    if args.localization:
        cmd.extend(["--localization", args.localization])
    else:
        # 使用 locale.getlocale()，这是 Python 3.13 推荐的、不会产生警告的新方法
        try:
            language_code = locale.getlocale()[0]
            if language_code and language_code.startswith("zh"):
                cmd.extend(["--localization", "zh-Hans"])
        except Exception:
            # 如果获取地区失败，就跳过，不影响主程序
            pass

    subprocess.Popen(cmd)


def launch(args):
    """主启动函数"""
    log.info("Starting SD-Trainer Mikazuki GUI...")
    log.info(f"Base directory: {base_dir_path()}, Working directory: {os.getcwd()}")
    log.info(f"{platform.system()} Python {platform.python_version()} {sys.executable}")

    # 我们已经手动处理了环境，所以这部分代码可以保持注释状态
    # if not args.skip_prepare_environment:
    #     prepare_environment(disable_auto_mirror=args.disable_auto_mirror)

    if not check_port_avaliable(args.port):
        avaliable = find_avaliable_ports(30000, 30000 + 20)
        if avaliable:
            log.info(f"Port {args.port} is not available, switching to {avaliable}")
            args.port = avaliable
        else:
            log.error("Default port is not available and could not find a fallback port.")
            return  # 端口找不到就直接退出

    # 优雅地处理 git 版本号，如果找不到就给一个默认值，避免日志里出现 fatal error
    try:
        version = git_tag(base_dir_path())
    except Exception:
        version = "dev (no tags found)"
    log.info(f"SD-Trainer Version: {version}")

    # --- 环境变量设置保持不变 ---
    os.environ["MIKAZUKI_HOST"] = args.host
    os.environ["MIKAZUKI_PORT"] = str(args.port)
    os.environ["MIKAZUKI_TENSORBOARD_HOST"] = args.tensorboard_host
    os.environ["MIKAZUKI_TENSORBOARD_PORT"] = str(args.tensorboard_port)
    os.environ["MIKAZUKI_DEV"] = "1" if args.dev else "0"
    # ----------------------------------------------------------------------------------
    # 【新增代码】: 启用 PyTorch Dynamo 加速
    # 设置 accelerate 使用 inductor 后端，这会显著提升训练速度
    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "inductor"
    log.info("Enabled PyTorch Dynamo optimization with backend: inductor")
    # ----------------------------------------------------------------------------------
    if args.listen:
        args.host = "0.0.0.0"
        args.tensorboard_host = "0.0.0.0"

    # --- 启动子进程，并将 args 传递进去 ---
    if not args.disable_tageditor:
        run_tag_editor(args)

    if not args.disable_tensorboard:
        run_tensorboard(args)

    # --- 启动主服务 ---
    import uvicorn
    log.info(f"Server started at http://{args.host}:{args.port}")
    uvicorn.run("mikazuki.app:app", host=args.host, port=args.port, log_level="error", reload=args.dev)


if __name__ == "__main__":
    # 将解析出的 args 作为参数传递给 launch 函数，而不是作为全局变量使用
    args, _ = parser.parse_known_args()
    launch(args)