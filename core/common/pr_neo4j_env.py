import os
import threading
from typing import Any, Optional

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings

warnings.filterwarnings("ignore")

"""
说明：
- Neo4j 连接改为惰性初始化：仅在真正执行查询时尝试连接，避免 import 阶段报错。
- 凭据优先从环境变量读取，避免在代码中硬编码敏感信息。
"""

# 基础默认值仅保留非敏感项；密码默认留空，要求通过环境变量注入。
DEFAULT_NEO4J_URI = "neo4j://127.0.0.1:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = ""
DEFAULT_NEO4J_DATABASE = "neo4j"

NEO4J_URI = os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI)
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", DEFAULT_NEO4J_USERNAME)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", DEFAULT_NEO4J_DATABASE)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1") + "/embeddings"

# 公关传播RAG配置参数（v1.1 新结构）
PR_NODE_TYPES = {
    'CategoryL1': '一级分类节点',
    'CategoryL2': '二级分类节点',
    'Section': '内容分段节点',
    'Company': '公司实体节点',
    'Brand': '品牌实体节点',
    'CompanyType': '组织类型节点',
    'Campaign': '传播活动节点',
    'Concept': '概念/主题节点'
}

PR_RELATIONSHIPS = {
    'HAS_SUBCATEGORY': '一级分类包含二级分类',
    'HAS_SECTION': '二级分类包含内容分段',
    'MENTIONS_COMPANY': '内容分段提到公司',
    'MENTIONS_BRAND': '内容分段提到品牌',
    'INVOLVED_IN_CATEGORY': '公司参与分类',
    'BELONGS_TO_BRAND': '公司隶属于品牌',
    'BELONGS_TO_TYPE': '公司属于组织类型',
    'OPERATES_IN_TYPE': '品牌关联组织类型',
    'SPO_REL': '公司语义行为关系'
}

# v1.1 默认不依赖旧向量索引；若需要向量检索，请根据 Section 节点自建索引
VECTOR_INDEX_NAME = os.getenv('SECTION_VECTOR_INDEX', 'SectionEmbedding')
VECTOR_NODE_LABEL = os.getenv('SECTION_VECTOR_LABEL', 'Section')
VECTOR_SOURCE_PROPERTY = os.getenv('SECTION_VECTOR_SOURCE_PROP', 'content')
VECTOR_EMBEDDING_PROPERTY = os.getenv('SECTION_VECTOR_EMBED_PROP', 'textEmbedding')

# 公关传播特定属性
PR_PROPERTIES = {
    'Brand': ['name', 'industry', 'founded_year', 'brand_value'],
    'Agency': ['name', 'founded_year', 'specialization', 'reputation'],
    'Campaign': ['name', 'launch_date', 'budget', 'duration', 'status'],
    'Strategy': ['strategy_type', 'target_audience', 'key_message', 'channels'],
    'Media': ['media_type', 'reach', 'engagement_rate', 'cost'],
    'Target_Audience': ['demographics', 'psychographics', 'behavior', 'size'],
    'Content': ['content_type', 'tone', 'format', 'performance'],
    'KPI': ['metric_name', 'target_value', 'actual_value', 'measurement_date']
}

_graph_instance: Optional[Neo4jGraph] = None
_graph_init_error: Optional[Exception] = None
_graph_lock = threading.Lock()


def _build_graph() -> Neo4jGraph:
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "缺少 NEO4J_PASSWORD。请在环境变量或 .env 中配置 NEO4J_PASSWORD 后再执行图谱查询。"
        )
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )


def get_graph() -> Optional[Neo4jGraph]:
    """惰性获取 Neo4jGraph；初始化失败时返回 None。"""
    global _graph_instance, _graph_init_error

    if _graph_instance is not None:
        return _graph_instance
    if _graph_init_error is not None:
        return None

    with _graph_lock:
        if _graph_instance is not None:
            return _graph_instance
        if _graph_init_error is not None:
            return None
        try:
            _graph_instance = _build_graph()
            return _graph_instance
        except Exception as exc:
            _graph_init_error = exc
            return None


def get_graph_error() -> Optional[Exception]:
    """返回最近一次初始化错误（若有）。"""
    return _graph_init_error


class _LazyGraph:
    """兼容旧代码中 `graph.query(...)` 的惰性代理。"""

    def _target(self) -> Neo4jGraph:
        target = get_graph()
        if target is None:
            err = get_graph_error()
            raise RuntimeError(f"Neo4j 图谱不可用: {err or '未初始化'}")
        return target

    def __bool__(self) -> bool:
        return get_graph() is not None

    def __getattr__(self, item: str) -> Any:
        return getattr(self._target(), item)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._target().query(*args, **kwargs)


graph = _LazyGraph()

