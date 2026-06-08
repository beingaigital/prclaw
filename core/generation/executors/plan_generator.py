"""
方案生成执行器。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.generation.postprocessors import format_plan_results
from core.generation.templates import PLAN_TEMPLATES, get_template

from .llm_executor import LLMExecutor


class PRPlanGenerator:
    """公关传播方案生成器（基于 v1.1 RAG 系统）。"""

    def __init__(self, rag_system=None, llm_config: Optional[Dict[str, Any]] = None) -> None:
        self.rag_system = rag_system
        base_config = llm_config or {}
        self.llm_config = {
            "provider": base_config.get("provider", "openai"),
            "model": base_config.get("model")  # 兼容旧字段
                     or base_config.get("flash_model")
                     or "gpt-4o-mini",
            "max_tokens": base_config.get("max_tokens", 2048),
            "temperature": base_config.get("temperature", 0.6),
            "fallback_providers": base_config.get("fallback_providers") or [],
            "max_retries": base_config.get("max_retries", 2),
            "retry_backoff_seconds": base_config.get("retry_backoff_seconds", 0.7),
        }
        self._executor = LLMExecutor(
            provider=self.llm_config["provider"],
            model=self.llm_config["model"],
            max_tokens=self.llm_config["max_tokens"],
            temperature=self.llm_config["temperature"],
            fallback_providers=self.llm_config["fallback_providers"],
            max_retries=self.llm_config["max_retries"],
            retry_backoff_seconds=self.llm_config["retry_backoff_seconds"],
        )

    @staticmethod
    def _is_failure_text(text: str) -> bool:
        low = (text or "").strip().lower()
        return low.startswith("生成失败[") or low.startswith("生成失败:")

    @staticmethod
    def _ensure_structure(plan_type: str, content: str, enterprise_info: Dict[str, Any]) -> str:
        """对 A/B/C 做最小结构保底，降低模型输出过散导致的交付风险。"""
        text = str(content or "").strip()
        if not text or PRPlanGenerator._is_failure_text(text):
            return text

        if plan_type == "C":
            required = ["目标与KPI", "受众洞察", "传播节奏", "渠道策略", "预算", "风险"]
            hit = sum(1 for x in required if x in text)
            if hit >= 4:
                return text
            enterprise = str(enterprise_info.get("enterprise_name", "品牌方")).strip() or "品牌方"
            goal = str(enterprise_info.get("pr_goal", "提升品牌影响力")).strip() or "提升品牌影响力"
            return (
                text
                + "\n\n---\n"
                + "以下为结构补全（自动追加）：\n"
                + f"一、目标与KPI\n- 品牌：{enterprise}\n- 目标：{goal}\n- KPI：曝光、互动、留资、转化。\n"
                + "二、受众洞察\n- 核心人群、场景与触发点。\n"
                + "三、传播节奏\n- 预热、引爆、沉淀三阶段与里程碑。\n"
                + "四、渠道策略\n- 公域平台矩阵 + 私域承接路径。\n"
                + "五、预算分配\n- 按内容、媒介、执行、应急拆分。\n"
                + "六、风险预案\n- 舆情、执行、转化风险与应对机制。\n"
            )

        if plan_type == "B":
            required = ["分镜", "旁白", "时长", "CTA"]
            hit = sum(1 for x in required if x in text)
            if hit >= 3:
                return text
            return (
                text
                + "\n\n---\n"
                + "以下为脚本结构补全（自动追加）：\n"
                + "- 时长：建议 45-60 秒\n"
                + "- 分镜：开场 Hook / 冲突 / 解决 / 价值升华 / CTA\n"
                + "- 旁白：每段一句核心信息，避免口号堆叠\n"
                + "- CTA：明确报名路径与截止时间\n"
            )

        if plan_type == "A":
            required = ["创意主题", "视觉", "应用场景"]
            hit = sum(1 for x in required if x in text)
            if hit >= 2:
                return text
            return (
                text
                + "\n\n---\n"
                + "以下为简报结构补全（自动追加）：\n"
                + "- 创意主题：一句话主张 + 传播口号\n"
                + "- 视觉规范：主色/辅助色/字体/版式比例\n"
                + "- 应用场景：KV海报、社媒封面、活动主视觉、校园物料\n"
            )

        return text

    def generate_plan(
        self,
        enterprise_info: Dict[str, Any],
        output_types: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成多种类型的方案。"""
        output_types = output_types or list(PLAN_TEMPLATES.keys())

        enriched_context = context or self._fetch_context(enterprise_info)
        if not enriched_context:
            enriched_context = "基于行业最佳实践和案例经验"

        vars_text = json.dumps(enterprise_info, ensure_ascii=False)

        results: Dict[str, Any] = {}
        for plan_type in output_types:
            template = get_template(plan_type)
            if not template:
                continue
            prompt = template.format(context=enriched_context, vars=vars_text)
            raw = self._executor.complete(prompt)
            results[plan_type] = self._ensure_structure(plan_type, raw, enterprise_info)

        return format_plan_results(results)

    def _fetch_context(self, enterprise_info: Dict[str, Any]) -> Optional[str]:
        """根据企业信息构建检索问题并调用 RAG。"""
        if not self.rag_system:
            return None
        query = self._build_query(enterprise_info)
        return self.rag_system.query(query, use_graph=True)

    @staticmethod
    def _build_query(enterprise_info: Dict[str, Any]) -> str:
        """构建查询语句。"""
        parts = []
        if enterprise_info.get("enterprise_stage"):
            parts.append(enterprise_info["enterprise_stage"])
        if enterprise_info.get("industry"):
            parts.append(enterprise_info["industry"])
        if enterprise_info.get("market_type"):
            parts.append(enterprise_info["market_type"])
        if enterprise_info.get("pr_goal"):
            parts.append(f"目标:{enterprise_info['pr_goal']}")
        if enterprise_info.get("innovation"):
            parts.append(f"创新:{enterprise_info['innovation']}")
        return " ".join(parts) if parts else "公关传播策略"
