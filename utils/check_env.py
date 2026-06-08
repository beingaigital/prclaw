"""启动前环境与关键配置自检（不发起 LLM 调用）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from utils.path import get_config_path, get_project_root


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        _warn(f"无法读取 {path}: {exc}")
        return {}


def _main_model_api_key_env() -> str:
    raw = _load_yaml(get_config_path("model.yaml"))
    main = raw.get("main")
    if isinstance(main, dict):
        key = str(main.get("api_key_env", "")).strip()
        if key:
            return key
    return "APIKEY"


def _load_dotenv_if_present(env_path: Path) -> None:
    """自检阶段也加载 .env，避免出现“已配置但误告警”的情况。"""
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        return


def run_environment_check() -> int:
    """
    执行环境检查，返回进程退出码：0 表示可继续运行，1 表示存在阻塞性错误。
    """
    root = get_project_root()
    errors: List[str] = []
    warnings: List[str] = []

    _ok(f"项目根目录: {root}")

    # 关键配置文件
    required_files: Tuple[str, ...] = (
        "prclaw.yaml",
        "model.yaml",
        "unified_config.yaml",
        "tools.yaml",
    )
    for name in required_files:
        p = get_config_path(name)
        if not p.exists():
            errors.append(f"缺少配置文件: {p}")
        else:
            _ok(f"配置文件存在: config/{name}")

    # .env（可选但推荐）
    env_path = root / ".env"
    if env_path.exists():
        _load_dotenv_if_present(env_path)
        _ok(".env 已存在（将按 python-dotenv 规则在运行时加载）")
    else:
        warnings.append("未找到 .env，请复制 .env.example 为 .env 并填写密钥与数据库信息")

    # LLM：按 model.yaml 的 api_key_env 检查
    api_key_env = _main_model_api_key_env()
    if not os.getenv(api_key_env):
        warnings.append(
            f"未设置环境变量 {api_key_env}（Agent 主模型需要；可在 .env 中配置）"
        )
    else:
        _ok(f"已设置 {api_key_env}")

    # Neo4j（图谱能力）
    if not os.getenv("NEO4J_PASSWORD"):
        warnings.append("未设置 NEO4J_PASSWORD（图谱检索与部分 RAG 将无法连接 Neo4j）")
    else:
        _ok("已设置 NEO4J_PASSWORD")

    # 可选：外部检索
    if not os.getenv("SERPER_API_KEY"):
        warnings.append("未设置 SERPER_API_KEY（web_search 在 provider=auto/serper 时可能降级）")

    # Vault：避免机器绝对路径未改
    unified = _load_yaml(get_config_path("unified_config.yaml"))
    vault = unified.get("vault") if isinstance(unified.get("vault"), dict) else {}
    vault_enabled = bool(vault.get("enabled", False))
    vault_path = str(vault.get("path", "")).strip()
    if vault_enabled:
        if not vault_path:
            warnings.append("vault.enabled 为 true 但 path 为空，请设置相对路径或 VAULT_PATH")
        elif vault_path.startswith("/Users/") or vault_path.startswith("/home/"):
            _warn("vault.path 为绝对路径，换机部署时请改为相对路径或环境变量")
        vp = Path(vault_path).expanduser()
        if not vp.is_absolute():
            vp = (root / vp).resolve()
        if vault_path and not vp.exists():
            warnings.append(f"Vault 目录不存在: {vp}")
        else:
            _ok(f"Vault 路径可解析: {vp}")
    else:
        _ok("Vault 未启用（unified_config.yaml vault.enabled: false）")

    # 导入冒烟
    try:
        import cli.main  # noqa: F401
        import utils.unified_adapter  # noqa: F401
        _ok("核心模块可导入")
    except Exception as exc:
        errors.append(f"核心模块导入失败: {exc}")

    for w in warnings:
        _warn(w)
    for e in errors:
        _err(e)

    if errors:
        print("\n检查未通过：请先修复上述 [error] 项。", file=sys.stderr)
        return 1
    if warnings:
        print("\n检查完成：存在 [warn] 项，仍可启动但部分功能可能不可用。")
    else:
        print("\n检查完成：环境与配置就绪。")
    return 0
