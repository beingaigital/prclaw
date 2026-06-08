"""
通用 LLM 执行器。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from core.common.llm_provider import classify_llm_error, get_chat_llm, resolve_llm_runtime


class LLMExecutor:
    """封装 ChatOpenAI 调用，便于统一配置与替换。"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.6,
        fallback_providers: Optional[List[str]] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.7,
    ) -> None:
        self.provider = provider
        self.model = model or "gpt-3.5-turbo"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_providers = fallback_providers or self._load_fallback_from_env()
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    @staticmethod
    def _load_fallback_from_env() -> List[str]:
        raw = os.getenv("LLM_FALLBACK_PROVIDERS", "")
        if not raw.strip():
            return []
        output: List[str] = []
        for item in raw.replace("，", ",").split(","):
            p = item.strip()
            if p and p not in output:
                output.append(p)
        return output

    def _candidate_providers(self) -> List[str]:
        providers = [self.provider]
        for p in self.fallback_providers:
            if p and p not in providers:
                providers.append(p)
        return providers

    def complete_with_meta(self, prompt: str) -> Dict[str, Any]:
        """
        执行一次补全，返回结构化结果，便于上层做可观测性和错误治理。
        """
        last_error_code = "E_LLM_UNKNOWN"
        last_error_message = "未知错误"
        attempts = 0

        for provider in self._candidate_providers():
            for retry_idx in range(self.max_retries):
                attempts += 1
                try:
                    llm = get_chat_llm(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        provider=provider,
                        tier="flash",
                    )
                    response = llm.invoke(prompt)
                    content = response.content if hasattr(response, "content") else str(response)
                    return {
                        "ok": "true",
                        "content": str(content or "").strip(),
                        "error_code": "",
                        "error_message": "",
                        "attempts": str(attempts),
                        "provider": provider,
                        "model": resolve_llm_runtime(model=self.model, provider=provider, tier="flash").model,
                    }
                except Exception as exc:  # pragma: no cover - 依赖外部服务
                    normalized = classify_llm_error(exc)
                    last_error_code = normalized["code"]
                    last_error_message = normalized["message"]
                    if retry_idx < self.max_retries - 1 and self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * (retry_idx + 1))
                    continue

        return {
            "ok": "false",
            "content": "",
            "error_code": last_error_code,
            "error_message": last_error_message,
            "attempts": str(attempts),
            "provider": self.provider,
            "model": self.model,
        }

    def complete(self, prompt: str) -> str:
        """执行一次补全。"""
        payload = self.complete_with_meta(prompt)
        if payload.get("ok") == "true":
            return payload.get("content", "")
        code = payload.get("error_code", "E_LLM_UNKNOWN")
        msg = payload.get("error_message", "未知错误")
        return f"生成失败[{code}]: {msg}"


def llm_complete(
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.6,
) -> str:
    """保持对旧接口的兼容。"""
    executor = LLMExecutor(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return executor.complete(prompt)
