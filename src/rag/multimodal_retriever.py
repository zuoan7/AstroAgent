"""
多模态检索模块

支持对天文图像、光谱数据的检索与理解:
  - 天文图像描述生成（基于 qwen-vl-plus 视觉模型）
  - 图像-文本跨模态嵌入
  - 光谱数据特征提取与检索
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


@dataclass
class MultimodalDocument:
    doc_id: str
    modality: str  # "text", "image", "spectrum"
    content: str = ""
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "doc_id": self.doc_id,
            "modality": self.modality,
            "metadata": self.metadata,
        }
        if self.content:
            d["content"] = self.content
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class MultimodalSearchResult:
    doc_id: str
    modality: str
    content: str
    description: Optional[str]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AstronomyImageDescriber:
    """天文图像描述生成器"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.model = model or settings.VISION_MODEL_NAME
        self.enabled = bool(self.api_key)

    def describe_image(self, image_input: str, prompt: str = "") -> str:
        """
        生成天文图像描述

        Args:
            image_input: 图片路径或 base64 编码
            prompt: 自定义提示词

        Returns:
            图像描述文本
        """
        if not self.enabled:
            return ""

        default_prompt = (
            "请详细描述这张天文图像中可见的天体、结构和特征。"
            "包括：天体类型、位置关系、颜色特征、可能的观测设备和波段信息。"
        )
        actual_prompt = prompt or default_prompt

        try:
            return self._call_vision_api(image_input, actual_prompt)
        except Exception as e:
            logger.error(f"❌ 天文图像描述生成失败: {e}")
            return ""

    def _call_vision_api(self, image_input: str, prompt: str) -> str:
        if os.path.isfile(image_input):
            with open(image_input, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_input)[1].lower()
            mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp", ".bmp": "bmp"}
            mime = mime_map.get(ext, "jpeg")
            image_url = f"data:image/{mime};base64,{image_b64}"
        elif image_input.startswith("data:"):
            image_url = image_input
        elif len(image_input) > 200:
            image_url = f"data:image/jpeg;base64,{image_input}"
        else:
            image_url = image_input

        try:
            import dashscope
            dashscope.api_key = self.api_key

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": prompt},
                    ],
                }
            ]

            resp = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
            )

            if resp.status_code == 200:
                output = resp.output
                choices = output.get("choices", []) if isinstance(output, dict) else []
                if choices:
                    content = choices[0].get("message", {}).get("content", [])
                    if isinstance(content, list):
                        text_parts = [c.get("text", "") for c in content if c.get("text")]
                        return " ".join(text_parts)
                    return str(content)
            return ""
        except ImportError:
            logger.warning("⚠️  dashscope SDK 未安装，尝试 HTTP API")
            return self._call_vision_http(image_url, prompt)
        except Exception as e:
            logger.error(f"❌ 视觉模型调用失败: {e}")
            return ""

    def _call_vision_http(self, image_url: str, prompt: str) -> str:
        try:
            import requests

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_url},
                            {"text": prompt},
                        ],
                    }
                ],
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            resp = requests.post(
                "https://dashscope.aliyuncs.com/compatible-api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"❌ 视觉模型 HTTP 调用失败: {e}")
            return ""


class SpectrumFeatureExtractor:
    """光谱数据特征提取器"""

    SPECTRAL_FEATURES = {
        "吸收线": re.compile(r'(Hα|Hβ|Hγ|Hδ|Ca.K|Na.D|巴耳末|Balmer|absorption)', re.IGNORECASE),
        "发射线": re.compile(r'(OIII|OII|NII|SII|Hα.*emit|emission.line|[OIII])', re.IGNORECASE),
        "连续谱": re.compile(r'(黑体|blackbody|连续谱|continuum|普朗克|Planck)', re.IGNORECASE),
        "红移": re.compile(r'(红移|redshift|z\s*=\s*[\d.]+)', re.IGNORECASE),
        "光谱型": re.compile(r'(O型|B型|A型|F型|G型|K型|M型|spectral.type.[OBAFGKM])', re.IGNORECASE),
    }

    def extract_features(self, text: str) -> Dict[str, Any]:
        features = {}
        for feature_name, pattern in self.SPECTRAL_FEATURES.items():
            matches = pattern.findall(text)
            if matches:
                features[feature_name] = matches
        return features

    def spectrum_similarity(self, query_features: Dict[str, Any], doc_features: Dict[str, Any]) -> float:
        if not query_features or not doc_features:
            return 0.0

        query_set = set()
        for feature_name, values in query_features.items():
            for v in values:
                query_set.add(f"{feature_name}:{str(v)}")

        doc_set = set()
        for feature_name, values in doc_features.items():
            for v in values:
                doc_set.add(f"{feature_name}:{str(v)}")

        if not query_set:
            return 0.0

        intersection = query_set & doc_set
        return len(intersection) / len(query_set)


class MultimodalRetriever:
    """多模态检索器"""

    def __init__(self):
        self.image_describer = AstronomyImageDescriber()
        self.spectrum_extractor = SpectrumFeatureExtractor()
        self.documents: List[MultimodalDocument] = []
        self._text_index: Dict[str, int] = {}
        self._spectrum_features: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc: MultimodalDocument) -> None:
        if doc.modality == "image" and not doc.description and doc.image_path:
            doc.description = self.image_describer.describe_image(doc.image_path)

        idx = len(self.documents)
        self.documents.append(doc)
        self._text_index[doc.doc_id] = idx

        if doc.modality == "spectrum":
            content = doc.content or doc.description or ""
            self._spectrum_features[doc.doc_id] = self.spectrum_extractor.extract_features(content)

    def add_documents(self, docs: List[MultimodalDocument]) -> None:
        for doc in docs:
            self.add_document(doc)

    def search(
        self,
        query: str,
        top_k: int = 5,
        modalities: Optional[List[str]] = None,
    ) -> List[MultimodalSearchResult]:
        if not self.documents:
            return []

        allowed = set(modalities) if modalities else {"text", "image", "spectrum"}

        query_spectrum_features = self.spectrum_extractor.extract_features(query)

        results = []
        for doc in self.documents:
            if doc.modality not in allowed:
                continue

            score = 0.0
            searchable_text = doc.content or ""

            if doc.description:
                searchable_text = (searchable_text + " " + doc.description).strip()

            if not searchable_text:
                continue

            query_lower = query.lower()
            for word in query_lower.split():
                if len(word) >= 2 and word in searchable_text.lower():
                    score += 0.1

            if doc.modality == "spectrum" and query_spectrum_features:
                doc_features = self._spectrum_features.get(doc.doc_id, {})
                spec_sim = self.spectrum_extractor.spectrum_similarity(query_spectrum_features, doc_features)
                score += spec_sim * 0.5

            if doc.metadata.get("celestial_object"):
                obj = doc.metadata["celestial_object"].lower()
                if obj in query_lower:
                    score += 0.3

            if score > 0:
                results.append(MultimodalSearchResult(
                    doc_id=doc.doc_id,
                    modality=doc.modality,
                    content=searchable_text,
                    description=doc.description,
                    score=score,
                    metadata=doc.metadata,
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_formatted(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.search(query, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "content": r.content,
                "metadata": {**r.metadata, "modality": r.modality},
                "score": r.score,
                "description": r.description,
            })
        return formatted
