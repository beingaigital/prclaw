"""Local file-backed knowledge retrieval for PR methodology and references."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.path import get_project_root


NO_HIT_MARKERS = (
    "❌ 未找到相关信息",
    "未找到相关信息",
    "查询失败",
    "Section检索失败",
    "回答生成失败",
)


def has_meaningful_answer(answer: str) -> bool:
    """Return whether a RAG answer contains usable retrieved context."""
    text = str(answer or "").strip()
    if not text:
        return False
    return not any(text.startswith(marker) for marker in NO_HIT_MARKERS)


def search_local_knowledge(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search local methodology rules and reference CSVs as a graph fallback."""
    terms = _extract_terms(query)
    results: List[Dict[str, Any]] = []
    results.extend(_search_methodology_rules(terms))
    results.extend(_search_reference_tables(terms))

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in sorted(results, key=lambda row: row.get("score", 0), reverse=True):
        key = (item.get("source"), item.get("id") or item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break

    answer = _format_results(deduped)
    return {
        "enabled": True,
        "hit": bool(deduped),
        "terms": terms,
        "answer": answer,
        "results": deduped,
    }


def _extract_terms(query: str) -> List[str]:
    normalized = str(query or "").lower()
    for phrase in ["是什么", "有哪些", "有什么", "请问", "一下", "以及", "如何", "怎么", "为什么"]:
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"[，。！？、；：,.!?;:（）()\[\]【】\"'“”‘’]", " ", normalized)
    normalized = normalized.replace("的", " ")

    stop_words = {
        "什么",
        "哪些",
        "原则",
        "问题",
        "方法",
        "策略",
        "进行",
        "提供",
        "需要",
        "应该",
    }
    terms: List[str] = []
    priority_phrases = [
        "危机公关",
        "危机应对",
        "品牌认知",
        "用户增长",
        "新品发布",
        "销售促进",
        "线索获取",
        "整合营销",
        "思想领导力",
    ]
    for phrase in priority_phrases:
        if phrase.lower() in normalized:
            terms.append(phrase.lower())

    for chunk in re.findall(r"[\u4e00-\u9fa5a-z0-9]+", normalized):
        if len(chunk) < 2 or chunk in stop_words:
            continue
        terms.append(chunk)
        if len(chunk) > 4:
            for phrase in priority_phrases:
                if phrase.lower() in chunk:
                    terms.append(phrase.lower())

    unique: List[str] = []
    seen = set()
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return unique[:12]


def _score_text(text: str, terms: Iterable[str]) -> int:
    haystack = text.lower()
    score = 0
    for term in terms:
        if term and term in haystack:
            score += max(4, len(term) * 2)
    return score


def _search_methodology_rules(terms: List[str]) -> List[Dict[str, Any]]:
    path = get_project_root() / "data" / "rlhf" / "methodology_rules.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    raw_rules = payload.get("rules", []) if isinstance(payload, dict) else []
    results: List[Dict[str, Any]] = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            continue
        fields = [
            rule.get("rule_id", ""),
            rule.get("name", ""),
            rule.get("description", ""),
            json.dumps(rule.get("conditions", {}), ensure_ascii=False),
            json.dumps(rule.get("application_scenarios", []), ensure_ascii=False),
            json.dumps(rule.get("effects", {}), ensure_ascii=False),
            rule.get("content", ""),
        ]
        text = "\n".join(str(item) for item in fields if item)
        score = _score_text(text, terms)
        if score < 4:
            continue
        results.append(
            {
                "source_type": "methodology_rule",
                "source": "data/rlhf/methodology_rules.json",
                "id": rule.get("rule_id", ""),
                "title": rule.get("name", ""),
                "description": rule.get("description", ""),
                "conditions": rule.get("conditions", {}),
                "effects": rule.get("effects", {}),
                "content": rule.get("content", ""),
                "priority": rule.get("priority", 0),
                "score": score + int(rule.get("priority", 0) or 0) // 10,
            }
        )
    return results


def _search_reference_tables(terms: List[str]) -> List[Dict[str, Any]]:
    ref_dir = get_project_root() / "data" / "reference"
    if not ref_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for path in sorted(ref_dir.glob("*_表格.csv")):
        for index, row in enumerate(_read_csv_rows(path), start=1):
            text = " ".join(str(value) for value in row.values() if str(value).strip())
            score = _score_text(text, terms)
            if score < 4:
                continue
            useful_fields = {
                key: value
                for key, value in row.items()
                if str(value).strip() and str(value).strip().lower() != "nan"
            }
            title = _first_non_empty(
                useful_fields,
                ["品牌/项目", "企业/品牌", "二级分类", "一级分类", "一级行业分类", "二级行业分类"],
            )
            results.append(
                {
                    "source_type": "reference_table",
                    "source": f"data/reference/{path.name}",
                    "id": str(index),
                    "title": title or f"{path.stem} 第 {index} 行",
                    "fields": dict(list(useful_fields.items())[:8]),
                    "score": score,
                }
            )
    return results


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    for encoding in ("utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                return list(csv.DictReader(fh))
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return []


def _first_non_empty(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    lines = [f"本地知识库召回 {len(results)} 条:"]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item.get('title') or item.get('id')}")
        if item.get("description"):
            lines.append(f"   - 说明: {item['description']}")
        if item.get("conditions"):
            lines.append(f"   - 适用条件: {json.dumps(item['conditions'], ensure_ascii=False)}")
        if item.get("effects"):
            lines.append(f"   - 执行动作: {json.dumps(item['effects'], ensure_ascii=False)}")
        if item.get("content"):
            lines.append(f"   - 内容: {item['content']}")
        if item.get("fields"):
            field_text = "；".join(f"{key}: {value}" for key, value in item["fields"].items())
            lines.append(f"   - 表格字段: {field_text}")
        source = item.get("source", "")
        source_id = item.get("id", "")
        lines.append(f"   - 来源: {source}{('#' + source_id) if source_id else ''}")
    return "\n".join(lines)
