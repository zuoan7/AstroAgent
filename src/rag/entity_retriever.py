"""
天文专业实体检索器

基于天文领域实体词典实现第三路检索:
  - 天体名称匹配（行星、恒星、星系、星云、星团）
  - 天文术语匹配（坐标系统、星等、光谱型等）
  - 观测设备匹配（望远镜、探测器等）
  - 天文事件匹配（日食、月食、流星雨等）

与向量检索、BM25检索构成三路混合检索
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.config import settings
from src.core.logger import logger


PLANETS = {
    "水星": "mercury", "金星": "venus", "火星": "mars",
    "木星": "jupiter", "土星": "saturn", "天王星": "uranus", "海王星": "neptune",
    "冥王星": "pluto",
    "mercury": "mercury", "venus": "venus", "mars": "mars",
    "jupiter": "jupiter", "saturn": "saturn", "uranus": "uranus",
    "neptune": "neptune", "pluto": "pluto",
}

BRIGHT_STARS = {
    "天狼星": "sirius", "参宿四": "betelgeuse", "织女星": "vega",
    "大角星": "arcturus", "五车二": "capella", "参宿七": "rigel",
    "南河三": "procyon", "河鼓二": "altair", "心宿二": "antares",
    "北极星": "polaris", "角宿一": "spica",
    "sirius": "sirius", "betelgeuse": "betelgeuse", "vega": "vega",
    "arcturus": "arcturus", "capella": "capella", "rigel": "rigel",
    "procyon": "procyon", "altair": "altair", "antares": "antares",
    "polaris": "polaris", "spica": "spica",
}

DEEP_SKY_OBJECTS = {
    "仙女座星系": "M31", "仙女座大星系": "M31", "猎户座星云": "M42",
    "猎户座大星云": "M42", "蟹状星云": "M1", "环状星云": "M57",
    "昴星团": "M45", "毕星团": "hyades", "草帽星系": "M104",
    "旋涡星系": "M51", "鹰状星云": "M16", "礁湖星云": "M8",
    "三叶星云": "M20", "玫瑰星云": "ngc2237",
    "andromeda": "M31", "orion.nebula": "M42",
    "crab.nebula": "M1", "ring.nebula": "M57",
    "pleiades": "M45", "sombrero": "M104",
    "whirlpool": "M51", "eagle.nebula": "M16",
}

CONSTELLATIONS = {
    "仙女座": "andromeda", "猎户座": "orion", "大熊座": "ursa.major",
    "小熊座": "ursa.minor", "天琴座": "lyra", "天鹅座": "cygnus",
    "天鹰座": "aquila", "天蝎座": "scorpius", "射手座": "sagittarius",
    "狮子座": "leo", "双子座": "gemini", "金牛座": "taurus",
    "处女座": "virgo", "白羊座": "aries", "天秤座": "libra",
    "andromeda": "andromeda", "orion": "orion", "ursa.major": "ursa.major",
    "lyra": "lyra", "cygnus": "cygnus", "scorpius": "scorpius",
    "sagittarius": "sagittarius", "leo": "leo",
}

ASTRONOMY_TERMS = {
    "赤经": "right_ascension", "赤纬": "declination",
    "视星等": "apparent_magnitude", "绝对星等": "absolute_magnitude",
    "光年": "light_year", "秒差距": "parsec",
    "红移": "redshift", "蓝移": "blueshift",
    "光谱型": "spectral_type", "色指数": "color_index",
    "自行": "proper_motion", "视差": "parallax",
    "光变曲线": "light_curve", "径向速度": "radial_velocity",
    "吸积盘": "accretion_disk", "事件视界": "event_horizon",
    "中子星": "neutron_star", "黑洞": "black_hole",
    "白矮星": "white_dwarf", "红巨星": "red_giant",
    "超新星": "supernova", "变星": "variable_star",
    "双星": "binary_star", "脉冲星": "pulsar",
    "类星体": "quasar", "暗物质": "dark_matter",
    "暗能量": "dark_energy", "宇宙微波背景": "cmb",
    "right.ascension": "right_ascension", "declination": "declination",
    "magnitude": "apparent_magnitude", "light.year": "light_year",
    "parsec": "parsec", "redshift": "redshift",
    "spectral.type": "spectral_type", "accretion.disk": "accretion_disk",
    "black.hole": "black_hole", "neutron.star": "neutron_star",
    "white.dwarf": "white_dwarf", "red.giant": "red_giant",
    "supernova": "supernova", "pulsar": "pulsar",
    "quasar": "quasar", "dark.matter": "dark_matter",
}

OBSERVING_EVENTS = {
    "日食": "solar_eclipse", "月食": "lunar_eclipse",
    "流星雨": "meteor_shower", "彗星": "comet",
    "冲日": "opposition", "合日": "conjunction",
    "凌日": "transit", "掩星": "occultation",
    "极光": "aurora", "超级月亮": "supermoon",
    "solar.eclipse": "solar_eclipse", "lunar.eclipse": "lunar_eclipse",
    "meteor.shower": "meteor_shower", "comet": "comet",
    "opposition": "opposition", "conjunction": "conjunction",
    "transit": "transit", "occultation": "occultation",
    "aurora": "aurora", "supermoon": "supermoon",
}

INSTRUMENTS = {
    "哈勃望远镜": "hst", "韦伯望远镜": "jwst",
    "阿塔卡马": "alma", "甚大望远镜": "vlt",
    "凯克望远镜": "keck", "昴星望远镜": "subaru",
    "钱德拉": "chandra", "盖亚": "gaia",
    "hst": "hst", "jwst": "jwst", "alma": "alma",
    "vlt": "vlt", "keck": "keck", "chandra": "chandra",
    "gaia": "gaia", "tess": "tess",
}

ALL_ENTITY_DICTS = {
    "planet": PLANETS,
    "star": BRIGHT_STARS,
    "deep_sky": DEEP_SKY_OBJECTS,
    "constellation": CONSTELLATIONS,
    "term": ASTRONOMY_TERMS,
    "event": OBSERVING_EVENTS,
    "instrument": INSTRUMENTS,
}


@dataclass
class EntityMatch:
    entity: str
    canonical: str
    category: str
    start: int = 0
    end: int = 0
    confidence: float = 1.0


@dataclass
class EntitySearchResult:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    matched_entities: List[EntityMatch] = field(default_factory=list)


class AstronomyEntityRecognizer:
    """天文实体识别器"""

    def __init__(self):
        self._entity_patterns: List[Tuple[str, str, re.Pattern]] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        for category, entity_dict in ALL_ENTITY_DICTS.items():
            sorted_entities = sorted(entity_dict.keys(), key=len, reverse=True)
            for entity_name in sorted_entities:
                canonical = entity_dict[entity_name]
                pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
                self._entity_patterns.append((category, canonical, pattern))

    def recognize(self, text: str) -> List[EntityMatch]:
        matches = []
        for category, canonical, pattern in self._entity_patterns:
            for m in pattern.finditer(text):
                matches.append(EntityMatch(
                    entity=m.group(0),
                    canonical=canonical,
                    category=category,
                    start=m.start(),
                    end=m.end(),
                ))

        seen = set()
        unique = []
        for match in matches:
            key = (match.canonical, match.start, match.end)
            if key not in seen:
                seen.add(key)
                unique.append(match)

        unique.sort(key=lambda x: x.start)
        return unique


class AstronomyEntityRetriever:
    """天文实体检索器 - 第三路检索"""

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self.recognizer = AstronomyEntityRecognizer()
        self.documents: List[Dict[str, Any]] = []
        self.entity_index: Dict[str, List[int]] = {}

        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        self.entity_index = {}

        for i, doc in enumerate(documents):
            content = doc.get("content", "") or doc.get("document", "")
            if not content:
                continue

            entities = self.recognizer.recognize(content)
            for entity in entities:
                key = entity.canonical
                if key not in self.entity_index:
                    self.entity_index[key] = []
                self.entity_index[key].append(i)

            doc["_entities"] = entities

        logger.info(f"✅ 天文实体索引构建完成: {len(self.documents)} 个文档, {len(self.entity_index)} 个实体")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.1,
    ) -> List[EntitySearchResult]:
        if not self.documents or not self.entity_index:
            return []

        query_entities = self.recognizer.recognize(query)
        if not query_entities:
            return []

        query_canonicals = set()
        for e in query_entities:
            query_canonicals.add(e.canonical)

        doc_scores: Dict[int, float] = {}
        doc_matched: Dict[int, List[EntityMatch]] = {}

        for canonical in query_canonicals:
            if canonical not in self.entity_index:
                related = self._find_related(canonical)
                for rel in related:
                    if rel in self.entity_index:
                        for idx in self.entity_index[rel]:
                            doc_scores[idx] = doc_scores.get(idx, 0) + 0.5
                continue

            for idx in self.entity_index[canonical]:
                doc_scores[idx] = doc_scores.get(idx, 0) + 1.0
                if idx not in doc_matched:
                    doc_matched[idx] = []
                doc_matched[idx].append(
                    EntityMatch(entity=canonical, canonical=canonical, category="matched")
                )

        max_score = max(doc_scores.values()) if doc_scores else 1.0
        results = []
        for idx, score in doc_scores.items():
            normalized = score / max_score
            if normalized < min_score:
                continue
            doc = self.documents[idx]
            content = doc.get("content", "") or doc.get("document", "")
            results.append(EntitySearchResult(
                content=content,
                metadata=doc.get("metadata", {}),
                score=normalized,
                matched_entities=doc_matched.get(idx, []),
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        logger.info(f"📄 天文实体检索返回 {len(results)} 个结果 (查询实体: {len(query_entities)})")
        return results

    def _find_related(self, canonical: str) -> List[str]:
        related = []
        for category, entity_dict in ALL_ENTITY_DICTS.items():
            if canonical in entity_dict.values():
                for name, canon in entity_dict.items():
                    if canon == canonical and name != canonical:
                        related.append(canon)
        return related

    def search_formatted(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.search(query, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
                "matched_entities": [
                    {"entity": m.entity, "canonical": m.canonical, "category": m.category}
                    for m in r.matched_entities
                ],
            })
        return formatted
