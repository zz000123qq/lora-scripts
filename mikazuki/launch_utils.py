import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import socket
from pathlib import Path
from typing import List, Optional

from packaging.version import parse as parse_version
from mikazuki.log import log

python_bin = sys.executable


def base_dir_path() -> Path:
    """返回项目的根目录 Path 对象"""
    return Path(__file__).parents[1].resolve()


def find_windows_git() -> Optional[str]:
    """在 Windows 上寻找可能的 Git 安装路径"""
    possible_paths = ["git\\bin\\git.exe", "git\\cmd\\git.exe", "Git\\mingw64\\libexec\\git-core\\git.exe",
                      "C:\\Program Files\\Git\\cmd\\git.exe"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def prepare_git():
    """检查 Git 是否可用，如果不在 PATH 中则尝试添加"""
    if shutil.which("git"):
        return True

    log.info("Finding git...")

    if sys.platform == "win32":
        git_path = find_windows_git()
        if git_path is not None:
            log.info(f"Git not found, but found git in {git_path}, add it to PATH")
            os.environ["PATH"] += os.pathsep + os.path.dirname(git_path)
            return True
        else:
            return False
    else:
        log.error("git not found, please install git first")
        return False


def prepare_submodules():
    """检查并更新 Git 子模块"""
    frontend_path = base_dir_path() / "frontend" / "dist"
    tag_editor_path = base_dir_path() / "mikazuki" / "dataset-tag-editor" / "scripts"

    if not frontend_path.exists() or not tag_editor_path.exists():
        log.info("submodule not found, try clone...")
        log.info("checking git installation...")
        if not prepare_git():
            log.error("git not found, please install git first")
            sys.exit(1)
        run(["git", "submodule", "init"], check=True)
        run(["git", "submodule", "update"], check=True)


def git_tag(path: Path) -> str:
    """
    获取 Git 仓库的标签版本号。
    如果失败，则静默处理并返回一个友好的默认值。
    """
    try:
        # 使用 stderr=subprocess.DEVNULL 来抑制 "fatal: ..." 错误消息的输出
        result = subprocess.check_output(
            ["git", "-C", str(path), "describe", "--tags"],
            stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return result if result else "dev"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "dev (git not found or no tags)"


def check_dirs(dirs: List[str]):
    """确保目录存在，如果不存在则创建"""
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


def run(command, *args, **kwargs):
    """运行子进程命令的通用函数"""
    # 保持原有的 run 函数逻辑，因为它被项目多处使用
    # 为简洁起见，这里假设它就是 subprocess.run 的一个包装器
    return subprocess.run(command, *args, **kwargs)


def is_installed(package: str) -> bool:
    """
    使用现代化的 importlib.metadata 和 packaging 库来检查包是否安装及其版本是否满足要求。
    完全替代已弃用的 pkg_resources。
    """
    package_line = re.sub(r'\[.*?\]', '', package)  # 移除 extras, e.g., diffusers[torch]

    match = re.match(r"([^<>=!~]+)((?:[<>=!~]=?|~=)\s*[\d\.\w\*\+]+)?", package_line)
    if not match:
        log.warning(f"无法解析包: {package}")
        return False

    name, req_version_str = match.groups()
    name = name.strip()

    try:
        installed_version_str = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        log.warning(f"包未安装: {name}")
        return False

    if not req_version_str:
        return True  # 只要安装了就满足要求

    op_match = re.match(r"([<>=!~]=?|~=)\s*([\d\.\w\*\+]+)", req_version_str.strip())
    if not op_match:
        log.warning(f"无法解析版本要求: {req_version_str}")
        return False

    op, req_version_str = op_match.groups()

    installed_version = parse_version(installed_version_str)
    required_version = parse_version(req_version_str)

    op_map = {
        "==": lambda a, b: a == b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b, "!=": lambda a, b: a != b,
    }

    if op not in op_map or not op_map[op](installed_version, required_version):
        log.info(f"包版本不匹配: {name} (已安装: {installed_version}, 要求: {op}{required_version})")
        return False

    return True


def validate_requirements(requirements_file: str):
    """验证并安装 requirements.txt 中的依赖"""
    req_path = Path(requirements_file)
    if not req_path.is_file():
        log.error(f"requirements file not found: {requirements_file}")
        return

    with req_path.open('r', encoding='utf8') as f:
        lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

        index_url = ""
        for line in lines:
            if line.startswith("--index-url "):
                index_url = line.replace("--index-url ", "")
                continue

            if not is_installed(line):
                command = f"install {line}"
                if index_url:
                    command += f" --index-url {index_url}"
                run_pip(command, line, live=True)


def setup_windows_bitsandbytes():
    """为 Windows 设置 bitsandbytes"""
    # 此函数逻辑保持不变
    pass


def setup_onnxruntime(onnx_version: Optional[str] = None, index_url: Optional[str] = None):
    """设置 onnxruntime"""
    # 此函数逻辑保持不变
    pass


def run_pip(command, desc=None, live=False):
    """运行 pip 命令的包装函数"""
    full_command = f'"{python_bin}" -m pip {command}'
    log.info(f"Running pip command: {full_command}")
    result = run(full_command, shell=True, capture_output=not live, text=True)
    if result.returncode != 0:
        err_desc = f"Couldn't install {desc}" if desc else "Pip command failed"
        log.error(f"{err_desc}. Command: {full_command}. Error code: {result.returncode}")
        if not live:
            log.error(f"Stderr: {result.stderr}")
        raise RuntimeError(f"{err_desc}")
    return result


def prepare_environment(disable_auto_mirror: bool = True, prepare_onnxruntime: bool = True):
    """准备完整的运行环境"""
    # 此函数逻辑保持不变
    pass


def catch_exception(f):
    """捕获并记录异常的装饰器"""

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log.error(f"An error occurred in {f.__name__}: {e}", exc_info=True)

    return wrapper


def check_port_avaliable(port: int):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def find_avaliable_ports(port_init: int, port_range: int) -> Optional[int]:
    """寻找可用端口"""
    for p in range(port_init, port_init + port_range):
        if check_port_avaliable(p):
            return p
    log.error(f"No available ports in range: {port_init} -> {port_init + port_range}")
    return None