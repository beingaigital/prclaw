import importlib
import json

from core.querying.local_knowledge import has_meaningful_answer, search_local_knowledge


def test_crisis_pr_query_hits_local_methodology_rule():
    payload = search_local_knowledge("危机公关的原则是什么", max_results=3)

    assert payload["hit"] is True
    assert "危机应对 24 小时原则" in payload["answer"]
    assert "m_rule_crisis_response" in payload["answer"]


def test_no_hit_marker_is_not_meaningful_answer():
    assert has_meaningful_answer("❌ 未找到相关信息") is False
    assert has_meaningful_answer("基于召回材料的回答") is True


def test_query_tool_falls_back_to_local_knowledge(monkeypatch):
    module = importlib.import_module("tools.pr_query_knowledge")

    class DummyAdapter:
        def query_knowledge(self, query, use_graph=True):
            return "❌ 未找到相关信息"

    monkeypatch.setattr(module, "get_unified_adapter", lambda: DummyAdapter())
    monkeypatch.setattr(
        module,
        "search_web",
        lambda query, max_results: {"provider": "disabled", "results": [], "error": "disabled"},
    )

    result = module.pr_query_knowledge.invoke(
        {
            "query": "危机公关的原则是什么",
            "use_graph_rag": True,
            "use_web_search": True,
        }
    )
    payload = json.loads(result)

    assert payload["rag"]["hit"] is False
    assert payload["local_knowledge"]["hit"] is True
    assert payload["retrieval"]["has_internal_context"] is True
    assert "未找到相关信息" not in payload["merged_context"]
    assert "危机应对 24 小时原则" in payload["merged_context"]
