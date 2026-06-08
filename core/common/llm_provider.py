"""
LLM provider with safe fallback to avoid dependency/runtime crashes.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class _EchoResponse:
    content: str


class EchoLLM:
    """Fallback LLM that echoes prompt for offline/demo usage."""

    def __init__(self, *_, **__):
        pass

    def invoke(self, prompt: Any) -> _EchoResponse:
        text = prompt if isinstance(prompt, str) else getattr(prompt, "to_string", lambda: str(prompt))()
        return _EchoResponse(content=f"[offline echo]\n{text}")


@dataclass
class LLMRuntimeSettings:
    provider: str
    model: str
    tier: str
    base_url: str
    api_key_env: str
    api_key_present: bool


@dataclass
class LLMProbeResult:
    ok: bool
    error_code: str
    message: str
    settings: Dict[str, Any]


_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "models": {"flash": "gpt-4o-mini", "thinking": "gpt-4o"},
    },
    "openai_compatible": {
        "api_key_env": "APIKEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "models": {"flash": "gpt-4o-mini", "thinking": "gpt-4o"},
    },
    "kimi": {
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
        "base_url": "https://api.moonshot.cn/v1",
        "models": {"flash": "moonshot-v1-auto", "thinking": "moonshot-v1-128k"},
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "models": {"flash": "deepseek-chat", "thinking": "deepseek-reasoner"},
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": {"flash": "qwen-turbo", "thinking": "qwen-plus"},
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "GOOGLE_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": {"flash": "gemini-1.5-flash", "thinking": "gemini-1.5-pro"},
    },
}


_PROVIDER_ALIASES: Dict[str, str] = {
    "openai_compatible": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "compatible": "openai_compatible",
}


def classify_llm_error(error: Any) -> Dict[str, str]:
    """将模型调用异常标准化为错误码，供上层流程统一处理。"""
    msg = str(error or "").strip()
    low = msg.lower()

    if not msg:
        return {"code": "E_LLM_UNKNOWN", "message": "未知模型错误"}
    if "api_key" in low or "api key" in low or "unauthorized" in low or "401" in low:
        return {"code": "E_LLM_AUTH", "message": msg}
    if "timeout" in low or "timed out" in low:
        return {"code": "E_LLM_TIMEOUT", "message": msg}
    if "rate limit" in low or "429" in low or "quota" in low:
        return {"code": "E_LLM_RATE_LIMIT", "message": msg}
    if "connection error" in low or "failed to resolve" in low or "name resolution" in low:
        return {"code": "E_LLM_CONNECTION", "message": msg}
    if "invalid_request_error" in low or "bad request" in low or "400" in low:
        return {"code": "E_LLM_BAD_REQUEST", "message": msg}
    if "not found" in low or "model" in low and "exist" in low:
        return {"code": "E_LLM_MODEL_NOT_FOUND", "message": msg}
    return {"code": "E_LLM_UNKNOWN", "message": msg}


def _resolve_provider(provider: str | None) -> str:
    env_provider = os.getenv("LLM_PROVIDER")
    raw = (provider or env_provider or "openai").strip().lower()
    resolved = _PROVIDER_ALIASES.get(raw, raw)
    if resolved not in _PROVIDER_CONFIG:
        return "openai"
    return resolved


def _resolve_model(provider: str, tier: str, override_model: str | None) -> str:
    # 强制覆盖（用于 OpenAI-compatible 网关场景，把代码里写死的 gpt-* 统一映射到可用模型）
    force_model = os.getenv("LLM_FORCE_MODEL")
    if force_model:
        return force_model
    # 分档覆盖：优先于调用方显式 model，便于统一接管老代码中的硬编码模型
    tier_key = "LLM_THINKING_MODEL" if (tier or "").lower() == "thinking" else "LLM_FLASH_MODEL"
    tier_model = os.getenv(tier_key)
    if tier_model:
        return tier_model
    # env override
    env_model = os.getenv("LLM_MODEL")
    if override_model:
        return override_model
    if env_model:
        return env_model
    conf = _PROVIDER_CONFIG.get(provider, _PROVIDER_CONFIG["openai"])
    return conf["models"].get(tier, conf["models"]["flash"])


def _resolve_base_url(provider: str, default: str) -> str:
    env = _PROVIDER_CONFIG.get(provider, {}).get("base_url_env")
    if env and os.getenv(env):
        return os.getenv(env)
    return (
        os.getenv("LLM_BASE_URL")
        or os.getenv("API_BASE_URL")
        or os.getenv("BASE_URL")
        or default
    )


def _resolve_api_key(provider: str, default_env: str) -> str:
    env_key = os.getenv(default_env, "")
    if env_key:
        return env_key
    return (
        os.getenv("LLM_API_KEY", "")
        or os.getenv("APIKEY", "")
        or ""
    )


def resolve_llm_runtime(
    model: str | None = None,
    provider: str | None = None,
    tier: str = "flash",
) -> LLMRuntimeSettings:
    provider_resolved = _resolve_provider(provider)
    conf = _PROVIDER_CONFIG.get(provider_resolved, _PROVIDER_CONFIG["openai"])
    chosen_model = _resolve_model(provider_resolved, tier, model)
    base_url = _resolve_base_url(provider_resolved, conf["base_url"])
    api_key_env = str(conf.get("api_key_env", "") or "APIKEY")
    api_key = _resolve_api_key(provider_resolved, api_key_env)
    return LLMRuntimeSettings(
        provider=provider_resolved,
        model=chosen_model,
        tier=tier,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key_present=bool(api_key),
    )


def get_chat_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    provider: str | None = None,
    tier: str = "flash",
    allow_init_fallback: bool = True,
):
    """
    获取统一的 Chat LLM，支持 openai/kimi/deepseek/qwen/google，并按 tier（flash/thinking）选择模型。
    """
    runtime = resolve_llm_runtime(model=model, provider=provider, tier=tier)
    api_key = _resolve_api_key(runtime.provider, runtime.api_key_env)

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=runtime.model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or None,
            base_url=runtime.base_url,
        )
    except Exception as exc:  # pragma: no cover - fallback path
        if not allow_init_fallback:
            raise
        print(f"⚠️ ChatOpenAI 初始化失败，使用离线回显模型: {exc}")
        return EchoLLM()


def probe_chat_llm(
    model: str | None = None,
    provider: str | None = None,
    tier: str = "flash",
) -> Dict[str, Any]:
    """执行轻量连通性探针，返回标准化探测结果。"""
    runtime = resolve_llm_runtime(model=model, provider=provider, tier=tier)
    settings = asdict(runtime)

    if not runtime.api_key_present:
        result = LLMProbeResult(
            ok=False,
            error_code="E_LLM_MISSING_KEY",
            message=f"未检测到 {runtime.api_key_env}，无法进行 LLM 连通性探针",
            settings=settings,
        )
        return asdict(result)

    try:
        llm = get_chat_llm(
            model=model,
            provider=provider,
            tier=tier,
            temperature=0.0,
            max_tokens=16,
            allow_init_fallback=False,
        )
        _ = llm.invoke("Return exactly `OK`.")
        result = LLMProbeResult(
            ok=True,
            error_code="",
            message="LLM 连通性探针通过",
            settings=settings,
        )
        return asdict(result)
    except Exception as exc:
        normalized = classify_llm_error(exc)
        result = LLMProbeResult(
            ok=False,
            error_code=normalized["code"],
            message=normalized["message"],
            settings=settings,
        )
        return asdict(result)


__all__ = [
    "get_chat_llm",
    "resolve_llm_runtime",
    "probe_chat_llm",
    "classify_llm_error",
    "EchoLLM",
]
