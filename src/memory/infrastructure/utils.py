"""记忆基础设施通用工具函数。

集中放置 JSON 编解码、目录创建和有序去重等仓储层小工具，避免各仓储
重复实现细节。
"""

import json
import os
from datetime import datetime
from typing import Any, Optional, Sequence


def _json_dumps(value: Any) -> str:
    """用非 ASCII 转义关闭的方式序列化 JSON，保留中文可读性。"""

    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any) -> Any:
    """安全解析 JSON，空值或非法 JSON 返回调用方提供的默认值。"""

    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _utcnow_iso() -> str:
    return datetime.now().isoformat()


def _ensure_parent_dir(path: str):
    """确保数据库或备份文件所在目录存在。"""

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _dedupe_keep_order(items: Sequence[Any]) -> list[Any]:
    """按原顺序去重，可处理 dict/list 这类不可 hash 的值。"""

    seen = set()
    result: list[Any] = []
    for item in items:
        marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else item
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result
