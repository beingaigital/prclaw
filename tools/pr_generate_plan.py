"""公关传播方案生成工具。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import tool

from utils.unified_adapter import get_unified_adapter
from utils.path import get_project_root
from utils.plan_requirements_state import load_plan_requirements_state
from utils.prclaw_config import get_prclaw_config
from utils.task_context import get_task_id
from utils.web_search import format_web_context, search_web


PLAN_TYPE_LABELS: Dict[str, str] = {
    "A": "图文创意简报",
    "B": "视频脚本",
    "C": "整合活动方案",
    "D": "短视频脚本",
    "E": "小红书种草笔记",
    "F": "危机公关方案",
}

_ALLOWED_PLAN_TYPES = set(PLAN_TYPE_LABELS.keys())
_GENERATION_ERROR_HINTS = (
    "生成失败",
    "connection error",
    "apiconnectionerror",
    "timeout",
    "rate limit",
    "network",
    "invalid api key",
)


def _parse_output_types(raw: str, defaults: List[str]) -> List[str]:
    if not raw:
        return defaults
    chunks = raw.replace("，", ",").split(",")
    output: List[str] = []
    for chunk in chunks:
        key = chunk.strip().upper()
        if key in _ALLOWED_PLAN_TYPES and key not in output:
            output.append(key)
    return output or defaults


def _normalize_enterprise_info(data: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in data.items():
        normalized[key] = str(value or "").strip()
    return normalized


def _merge_enterprise_info(base: Dict[str, str], patch: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base)
    if not isinstance(patch, dict):
        return merged
    for key in base.keys():
        if not merged.get(key, "").strip():
            merged[key] = str(patch.get(key, "") or "").strip()
    return merged


def _looks_like_generation_error(content: Any) -> bool:
    text = str(content or "").strip().lower()
    if not text:
        return True
    return any(hint in text for hint in _GENERATION_ERROR_HINTS)


def _build_fallback_plan(output_type: str, enterprise_info: Dict[str, str], error_hint: str = "") -> str:
    enterprise = enterprise_info.get("enterprise_name") or "品牌方"
    industry = enterprise_info.get("industry") or "消费行业"
    goal = enterprise_info.get("pr_goal") or "提升品牌认知与参与度"
    cycle = enterprise_info.get("pr_cycle") or "8周"
    budget = enterprise_info.get("pr_budget") or "待确认"
    audience = enterprise_info.get("target_audience") or "18-24岁青年群体"
    messages = enterprise_info.get("key_messages") or "向美而声，青年共创"
    extra = enterprise_info.get("extra_requirements") or "线上线下联动"

    prefix = "说明：检测到模型连接异常，以下为系统自动生成的结构化兜底稿，可直接二次编辑。\n"
    if error_hint:
        prefix += f"异常摘要：{error_hint}\n\n"
    else:
        prefix += "\n"

    if output_type == "A":
        return (
            prefix
            + f"【图文创意简报｜{enterprise}】\n"
            + f"- 创意主题：向美而声，青年万有引力\n"
            + f"- 核心主张：围绕“{goal}”建立青年与品牌的双向吸引。\n"
            + "- 视觉调性：高对比青春色块 + 科技感网格 + 真人纪实特写。\n"
            + f"- 关键元素：青年人像、任务徽章、共创轨道、品牌资产（{messages}）。\n"
            + "- 适配场景：校园海报、地铁灯箱、社媒封面、活动签到背景板。\n"
            + f"- 执行提醒：人群聚焦“{audience}”，并确保素材可复用于短视频切片。\n"
        )

    if output_type == "B":
        return (
            prefix
            + f"【视频脚本｜{enterprise}】\n"
            + "时长：60秒\n"
            + "1. 0-5秒（Hook）：快节奏切镜呈现青年多元身份，字幕“年轻，不止一种定义”。\n"
            + f"2. 6-20秒（问题）：抛出青年成长焦虑与行业机会，点题“{industry}的新赛点”。\n"
            + "3. 21-40秒（解决）：展示项目机制（校园Fun卖会/黑客松/实习机会）与真实收益。\n"
            + f"4. 41-55秒（价值）：强化“{messages}”，呈现参与者成果与社会影响。\n"
            + "5. 56-60秒（CTA）：字幕“立即加入万有引力计划”，引导预约/报名。\n"
            + f"配音建议：年轻、坚定、真实；结尾落到“{goal}”。\n"
        )

    if output_type == "C":
        return (
            prefix
            + f"【整合活动方案｜{enterprise}】\n"
            + "一、目标与KPI\n"
            + f"- 传播目标：{goal}\n"
            + "- KPI建议：总曝光3000万+、互动率8%+、有效报名3000+、人才库沉淀1000+。\n"
            + "二、受众洞察\n"
            + f"- 核心人群：{audience}\n"
            + "- 关键洞察：青年希望“被看见、被赋能、可成长”，不只接受单向宣传。\n"
            + "三、核心叙事\n"
            + "- 战役口号：向美而声\n"
            + f"- 内容主轴：用“一人千面”的成长路径，承接“{messages}”。\n"
            + "四、传播节奏\n"
            + f"- 周期：{cycle}\n"
            + "- 阶段1 预热（25%）：校园话题测试 + 招募挑战赛\n"
            + "- 阶段2 引爆（45%）：主活动日 + KOL/KOC联动 + 直播共创\n"
            + "- 阶段3 沉淀（30%）：案例短片、成果展、雇主品牌内容回流\n"
            + "五、渠道策略\n"
            + "- 抖音/B站：挑战赛与纪录式内容\n"
            + "- 小红书/微博：参与心得与口碑扩散\n"
            + "- 微信私域：报名转化、社群运营与长期跟进\n"
            + f"- 创新玩法：{extra}\n"
            + "六、预算拆分\n"
            + f"- 总预算：{budget}\n"
            + "- 内容与创意 30% | 媒介投放 35% | 校园执行 20% | 数据与应急 15%\n"
            + "七、风险预案\n"
            + "- 负面舆情：24小时响应口径 + 双审批机制\n"
            + "- 进度偏差：按周复盘，保留10%机动预算用于加投或补救\n"
        )

    return (
        prefix
        + f"【{PLAN_TYPE_LABELS.get(output_type, output_type)}】\n"
        + f"- 品牌：{enterprise}\n"
        + f"- 行业：{industry}\n"
        + f"- 目标：{goal}\n"
        + f"- 周期：{cycle}\n"
        + f"- 预算：{budget}\n"
    )


def _render_markdown(
    plan_id: str,
    enterprise_info: Dict[str, str],
    output_types: List[str],
    plan_results: Dict[str, Any],
    context_query: str,
    web_payload: Dict[str, Any],
) -> str:
    lines = [
        f"# PRClaw 传播方案 - {enterprise_info.get('enterprise_name') or '未命名企业'}",
        "",
        f"- Plan ID: {plan_id}",
        f"- 生成时间: {datetime.now().isoformat()}",
        f"- 检索查询: {context_query}",
        "",
        "## 企业输入",
        "",
    ]

    for key in [
        "enterprise_name",
        "industry",
        "enterprise_stage",
        "market_type",
        "pr_goal",
        "pr_cycle",
        "pr_budget",
        "innovation",
        "target_audience",
        "key_messages",
        "extra_requirements",
    ]:
        value = enterprise_info.get(key, "")
        if value:
            lines.append(f"- {key}: {value}")

    web_items = web_payload.get("results") if isinstance(web_payload, dict) else []
    if isinstance(web_items, list) and web_items:
        lines.extend(["", "## 外部检索来源", ""])
        for idx, item in enumerate(web_items, start=1):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            lines.append(f"{idx}. {title} {url}".strip())

    lines.extend(["", "## 方案输出", ""])
    for output_type in output_types:
        label = PLAN_TYPE_LABELS.get(output_type, output_type)
        content = plan_results.get(output_type, "")
        lines.append(f"### {output_type} - {label}")
        lines.append("")
        lines.append(str(content).strip() if content else "（无输出）")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _save_markdown(content: str, enterprise_name: str, plan_id: str) -> str:
    output_dir = get_project_root() / "outputs" / "plans"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = enterprise_name.strip() or "enterprise"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe_name)
    file_path = output_dir / f"{plan_id}_{safe_name}.md"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@tool
def pr_generate_plan(
    enterprise_name: str,
    industry: str,
    pr_cycle: str,
    pr_budget: str,
    pr_goal: str,
    enterprise_stage: str = "",
    market_type: str = "",
    innovation: str = "适度创新",
    target_audience: str = "",
    key_messages: str = "",
    extra_requirements: str = "",
    output_types: str = "A,B,C",
    use_graph_rag: bool = True,
    use_web_search: bool = True,
    requirements_confirmed: bool = False,
    allow_fallback_template: bool = True,
) -> str:
    """
    描述：生成公关传播方案。输入企业信息后，自动融合内部 GraphRAG 知识与外部信息，输出多类型方案。
    使用时机：当用户明确要产出公关传播提案、campaign框架或传播执行方案。
    输入：
    - enterprise_name / industry / pr_cycle / pr_budget / pr_goal（必填）
    - enterprise_stage / market_type / innovation / target_audience / key_messages / extra_requirements（选填）
    - output_types（选填）：A-F，逗号分隔。
    - use_graph_rag / use_web_search（选填）：是否启用内部与外部检索。
    - requirements_confirmed（选填）：需求是否已与用户确认，默认 false。
    输出：JSON 字符串，包含方案结果、上下文信息、markdown 保存路径。
    """
    task_id = get_task_id()
    intake_state: Dict[str, Any] | None = None
    intake_req: Dict[str, Any] = {}
    intake_confirmed = False
    if task_id:
        try:
            intake_state = load_plan_requirements_state(task_id)
            intake_req = intake_state.get("requirements") if isinstance(intake_state.get("requirements"), dict) else {}
            intake_confirmed = bool(intake_state.get("confirmed"))
        except Exception:
            intake_state = None
            intake_req = {}
            intake_confirmed = False

    # 在会话模式中强制“先确认需求再生成”，避免模型猜测关键字段。
    if task_id and not intake_confirmed:
        gate_payload = {
            "error": "方案需求尚未确认，已拦截生成。",
            "need_confirmation": True,
            "next_action": "请先调用 pr_plan_requirements 收集并确认需求（用户回复“确认生成”后再调用本工具）。",
            "task_id": task_id,
            "current_requirements": intake_req,
        }
        return json.dumps(gate_payload, ensure_ascii=False)

    cfg = get_prclaw_config()
    adapter = get_unified_adapter()

    enterprise_info = _normalize_enterprise_info(
        {
            "enterprise_name": enterprise_name,
            "industry": industry,
            "pr_cycle": pr_cycle,
            "pr_budget": pr_budget,
            "pr_goal": pr_goal,
            "enterprise_stage": enterprise_stage,
            "market_type": market_type,
            "innovation": innovation,
            "target_audience": target_audience,
            "key_messages": key_messages,
            "extra_requirements": extra_requirements,
        }
    )

    if intake_req:
        enterprise_info = _merge_enterprise_info(enterprise_info, intake_req)

    final_output_types_raw = output_types
    intake_output_types = str(intake_req.get("output_types", "") or "").strip()
    if intake_output_types and (not output_types.strip() or output_types.strip().upper() == "A,B,C"):
        final_output_types_raw = intake_output_types

    selected_output_types = _parse_output_types(final_output_types_raw, cfg.default_output_types)
    context_query = adapter.build_plan_query(enterprise_info)

    rag_context = ""
    rag_error = ""
    if use_graph_rag:
        try:
            rag_context = adapter.query_knowledge(context_query, use_graph=True)
        except Exception as exc:
            rag_error = str(exc)

    web_payload = {"provider": "disabled", "results": [], "error": ""}
    if use_web_search:
        web_payload = search_web(query=context_query)

    context_parts: List[str] = []
    if rag_context:
        context_parts.append("内部知识库上下文:\n" + rag_context)
    web_context = format_web_context(web_payload)
    if web_context:
        context_parts.append(web_context)

    final_context = "\n\n".join(context_parts).strip() or None

    result = adapter.generate_plan(
        enterprise_info=enterprise_info,
        output_types=selected_output_types,
        context=final_context,
    )

    raw_generation_status = "success"
    failed_output_types: List[str] = []
    generation_errors: Dict[str, str] = {}
    fallback_used = False

    if not isinstance(result, dict):
        raw_generation_status = "failed"
        failed_output_types = list(selected_output_types)
        err_text = str(result).strip() or "未知生成错误"
        generation_errors = {k: err_text for k in selected_output_types}
        result = {}
    elif "error" in result:
        raw_generation_status = "failed"
        err_text = str(result.get("error", "")).strip() or "未知生成错误"
        failed_output_types = list(selected_output_types)
        generation_errors = {k: err_text for k in selected_output_types}
        result = {}
    else:
        for output_type in selected_output_types:
            content = result.get(output_type, "")
            if _looks_like_generation_error(content):
                failed_output_types.append(output_type)
                generation_errors[output_type] = str(content).strip()[:260] or "空内容"
        if failed_output_types:
            raw_generation_status = "failed" if len(failed_output_types) == len(selected_output_types) else "partial"

    if allow_fallback_template and failed_output_types:
        for output_type in failed_output_types:
            result[output_type] = _build_fallback_plan(
                output_type=output_type,
                enterprise_info=enterprise_info,
                error_hint=generation_errors.get(output_type, ""),
            )
        fallback_used = True

    final_status = raw_generation_status
    if fallback_used and raw_generation_status in {"failed", "partial"}:
        final_status = "success_with_fallback"

    plan_id = datetime.now().strftime("%Y%m%d%H%M%S")
    markdown_path = ""
    markdown_error = ""

    if isinstance(result, dict) and "error" not in result and cfg.export.save_markdown:
        try:
            markdown = _render_markdown(
                plan_id=plan_id,
                enterprise_info=enterprise_info,
                output_types=selected_output_types,
                plan_results=result,
                context_query=context_query,
                web_payload=web_payload,
            )
            markdown_path = _save_markdown(markdown, enterprise_info.get("enterprise_name", "enterprise"), plan_id)
        except Exception as exc:
            markdown_error = str(exc)

    payload = {
        "plan_id": plan_id,
        "status": final_status,
        "raw_generation_status": raw_generation_status,
        "fallback_used": fallback_used,
        "failed_output_types": failed_output_types,
        "generation_errors": generation_errors,
        "enterprise_info": enterprise_info,
        "output_types": selected_output_types,
        "result": result,
        "context": {
            "query": context_query,
            "graph_rag_enabled": bool(use_graph_rag),
            "graph_rag_error": rag_error,
            "web_enabled": bool(use_web_search),
            "web_provider": web_payload.get("provider", ""),
            "web_error": web_payload.get("error", ""),
            "web_results": web_payload.get("results", []),
        },
        "markdown_path": markdown_path,
        "markdown_error": markdown_error,
    }
    return json.dumps(payload, ensure_ascii=False)
