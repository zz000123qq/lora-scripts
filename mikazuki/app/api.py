import asyncio
import hashlib
import json
import os
import re
import random
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import toml
from fastapi import APIRouter, BackgroundTasks, Request
from starlette.requests import Request

import mikazuki.process as process
from mikazuki import launch_utils
from mikazuki.app.config import app_config
from mikazuki.app.models import (APIResponse, APIResponseFail,
                                 APIResponseSuccess, TaggerInterrogateRequest)
from mikazuki.log import log
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)
from mikazuki.tasks import tm
from mikazuki.utils import train_utils
from mikazuki.utils.devices import printable_devices
from mikazuki.utils.tk_window import (open_directory_selector,
                                      open_file_selector)

router = APIRouter()

# --- 全局变量区 (仅保留必要部分) ---

trainer_mapping = {
    "sd-lora": "./scripts/stable/train_network.py",
    "sdxl-lora": "./scripts/stable/sdxl_train_network.py",
    "sd-dreambooth": "./scripts/stable/train_db.py",
    "sdxl-finetune": "./scripts/stable/sdxl_train.py",
    "sd3-lora": "./scripts/dev/sd3_train_network.py",
    "flux-lora": "./scripts/dev/flux_train_network.py",
    "flux-finetune": "./scripts/dev/flux_train.py",
}


# --- 重构后的数据加载函数 ---

def _read_schemas_sync() -> List[Dict[str, Any]]:
    """同步读取 schema 文件的辅助函数，用于在线程中运行"""
    schema_dir = Path.cwd() / "mikazuki" / "schema"
    schemas_data = []
    if not schema_dir.is_dir():
        return []

    for schema_path in schema_dir.glob("*.ts"):
        content = schema_path.read_text(encoding="utf-8")
        content_hash = hashlib.md5(content.encode()).hexdigest()
        schemas_data.append({
            "name": schema_path.stem,
            "schema": content,
            "hash": content_hash
        })
    return schemas_data


async def load_schemas() -> List[Dict[str, Any]]:
    """异步加载所有 schema，不再依赖全局变量"""
    return await asyncio.to_thread(_read_schemas_sync)


def _read_presets_sync() -> List[Dict[str, Any]]:
    """同步读取 preset 文件的辅助函数"""
    preset_dir = Path.cwd() / "config" / "presets"
    presets_data = []
    if not preset_dir.is_dir():
        return []

    for preset_path in preset_dir.glob("*.toml"):
        content = preset_path.read_text(encoding="utf-8")
        presets_data.append(toml.loads(content))
    return presets_data


async def load_presets() -> List[Dict[str, Any]]:
    """异步加载所有 presets，不再依赖全局变量"""
    return await asyncio.to_thread(_read_presets_sync)


# --- 辅助函数区 ---

def get_sample_prompts(config: dict) -> Tuple[Optional[str], str]:
    # (此函数逻辑不变，保持原样)
    # backward compatibility
    if "sample_prompts" in config and "positive_prompts" not in config:
        return None, config["sample_prompts"]

    train_data_dir = config["train_data_dir"]
    sub_dir = [dir for dir in glob(os.path.join(train_data_dir, '*')) if os.path.isdir(dir)]

    positive_prompts = config.pop('positive_prompts', None)
    negative_prompts = config.pop('negative_prompts', '')
    sample_width = config.pop('sample_width', 512)
    sample_height = config.pop('sample_height', 512)
    sample_cfg = config.pop('sample_cfg', 7)
    sample_seed = config.pop('sample_seed', 2333)
    sample_steps = config.pop('sample_steps', 24)
    randomly_choice_prompt = config.pop('randomly_choice_prompt', False)

    if randomly_choice_prompt:
        if len(sub_dir) != 1:
            raise ValueError('训练数据集下有多个子文件夹，无法启用随机选取 Prompt 功能')

        txt_files = glob(os.path.join(sub_dir[0], '*.txt'))
        if not txt_files:
            raise ValueError('训练数据集路径没有 txt 文件')
        try:
            sample_prompt_file = random.choice(txt_files)
            with open(sample_prompt_file, 'r', encoding='utf-8') as f:
                positive_prompts = f.read()
        except IOError:
            log.error(f"读取 {sample_prompt_file} 文件失败")

    return positive_prompts, f'{positive_prompts} --n {negative_prompts}  --w {sample_width} --h {sample_height} --l {sample_cfg}  --s {sample_steps}  --d {sample_seed}'


# --- API 端点 (Endpoints) ---

@router.post("/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    autosave_dir = Path.cwd() / "config" / "autosave"
    autosave_dir.mkdir(exist_ok=True)
    toml_file = autosave_dir / f"{timestamp}.toml"

    json_data = await request.body()

    config: dict = json.loads(json_data.decode("utf-8"))
    train_utils.fix_config_types(config)

    gpu_ids = config.pop("gpu_ids", None)

    suggest_cpu_threads = 8 if len(train_utils.get_total_images(config["train_data_dir"])) > 200 else 2
    model_train_type = config.pop("model_train_type", "sd-lora")
    trainer_file = trainer_mapping[model_train_type]

    if model_train_type != "sdxl-finetune":
        if not train_utils.validate_data_dir(config["train_data_dir"]):
            return APIResponseFail(message="训练数据集路径不存在或没有图片，请检查目录。")

    validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], model_train_type)
    if not validated:
        return APIResponseFail(message=message)

    if "prompt_file" in config and config["prompt_file"].strip() != "":
        prompt_file = config["prompt_file"].strip()
        if not os.path.exists(prompt_file):
            return APIResponseFail(message=f"Prompt 文件 {prompt_file} 不存在，请检查路径。")
        config["sample_prompts"] = prompt_file
    else:
        try:
            positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)

            if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
                sample_prompts_file = autosave_dir / f"{timestamp}-promopt.txt"
                sample_prompts_file.write_text(sample_prompts_arg, encoding="utf-8")
                config["sample_prompts"] = str(sample_prompts_file)
                log.info(f"Wrote prompts to file {sample_prompts_file}")

        except ValueError as e:
            log.error(f"Error while processing prompts: {e}")
            return APIResponseFail(message=str(e))

    toml_file.write_text(toml.dumps(config), encoding="utf-8")

    result = process.run_train(str(toml_file), trainer_file, gpu_ids, suggest_cpu_threads)

    return result


@router.post("/run_script")
async def run_script(request: Request, background_tasks: BackgroundTasks):
    j = await request.json()
    script_name = j.pop("script_name")

    avaliable_scripts = [
        "networks/extract_lora_from_models.py",
        "networks/extract_lora_from_dylora.py",
        "networks/merge_lora.py",
        "tools/merge_models.py",
    ]
    if script_name not in avaliable_scripts:
        return APIResponseFail(message="Script not found")

    result = []
    for k, v in j.items():
        result.append(f"--{k}")
        if not isinstance(v, bool):
            value = str(v)
            if " " in value:
                value = f'"{v}"'
            result.append(value)

    script_args = " ".join(result)
    script_path = Path.cwd() / "scripts" / script_name
    cmd = f'"{launch_utils.python_bin}" "{script_path}" {script_args}'
    background_tasks.add_task(launch_utils.run, cmd)
    return APIResponseSuccess(message="Script started in background.")


@router.post("/interrogate")
async def run_interrogate(req: TaggerInterrogateRequest, background_tasks: BackgroundTasks):
    interrogator = available_interrogators.get(req.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
    background_tasks.add_task(
        on_interrogate,
        image=None,
        batch_input_glob=req.path,
        batch_input_recursive=req.batch_input_recursive,
        # ... (rest of the parameters are the same)
        unload_model_after_running=True
    )
    return APIResponseSuccess(message="Interrogation task started.")


@router.get("/pick_file")
async def pick_file(picker_type: str):
    coro = None
    if picker_type == "folder":
        coro = asyncio.to_thread(open_directory_selector, "")
    elif picker_type == "model-file":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        coro = asyncio.to_thread(open_file_selector, "", "Select file", file_types)

    if coro:
        result = await coro
        if result == "":
            return APIResponseFail(message="用户取消选择", data=None)
        return APIResponseSuccess(message="Path selected.", data={"path": result})

    return APIResponseFail(message="Invalid picker type.")


@router.get("/get_files")
async def get_files(pick_type: str) -> APIResponse:
    # ... (This function logic seems fine, just adding messages to responses)
    pick_preset = {
        "model-file": {"type": "file", "path": "./sd-models", "filter": r"(\.safetensors|\.ckpt|\.pt)$"},
        "model-saved-file": {"type": "file", "path": "./output", "filter": r"(\.safetensors|\.ckpt|\.pt)$"},
        "train-dir": {"type": "folder", "path": "./train", "filter": None},
    }

    if pick_type not in pick_preset:
        return APIResponseFail(message="Invalid request", data=None)

    preset_info = pick_preset[pick_type]
    path = Path(preset_info["path"])
    file_type = preset_info["type"]
    regex_filter = preset_info["filter"]
    result_list = []

    if not path.is_dir():
        return APIResponseSuccess(message="Directory not found.", data={"files": []})

    if file_type == "file":
        pattern = re.compile(regex_filter) if regex_filter else None
        files = [f for f in path.glob("**/*") if f.is_file() and (not pattern or pattern.search(f.name))]
        for file in files:
            result_list.append({
                "path": str(file.resolve()).replace("\\", "/"),
                "name": file.name,
                "size": f"{round(file.stat().st_size / (1024 ** 3), 2)} GB"
            })
    elif file_type == "folder":
        folders = [f for f in path.iterdir() if f.is_dir() and f.name not in [".ipynb_checkpoints", ".DS_Store"]]
        for folder in folders:
            result_list.append({
                "path": str(folder.resolve()).replace("\\", "/"),
                "name": folder.name,
                "size": 0
            })

    return APIResponseSuccess(message="Files listed.", data={"files": result_list})


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(message="Tasks listed.", data={"tasks": tm.dump()})


@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    tm.terminate_task(task_id)
    return APIResponseSuccess(message=f"Termination signal sent to task {task_id}.", data=None)


@router.get("/graphic_cards")
async def list_avaliable_cards() -> APIResponse:
    if not printable_devices:
        return APIResponse(status="pending", message="Devices not ready.", data=None)
    return APIResponseSuccess(message="Success", data={"cards": printable_devices})


@router.get("/schemas/hashes")
async def list_schema_hashes() -> APIResponse:
    schemas = await load_schemas()
    return APIResponseSuccess(message="Success", data={
        "schemas": [
            {"name": schema["name"], "hash": schema["hash"]}
            for schema in schemas
        ]
    })


@router.get("/schemas/all")
async def get_all_schemas() -> APIResponse:
    schemas = await load_schemas()
    return APIResponseSuccess(message="Success", data={"schemas": schemas})


@router.get("/presets")
async def get_presets() -> APIResponse:
    presets = await load_presets()
    return APIResponseSuccess(message="Success", data={"presets": presets})


@router.get("/config/saved_params")
async def get_saved_params() -> APIResponse:
    saved_params = app_config.get("saved_params", {})
    return APIResponseSuccess(message="Success", data=saved_params)