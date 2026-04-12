"""
天文领域专用文档分块器

根据天文数据特点设计专用分块逻辑:
  - 星图数据: 保持坐标系统完整性
  - 轨道参数: 保持参数组完整性（六根数）
  - 天体物理公式: 保持公式与上下文不分离
  - 观测记录: 按观测事件分块
  - 星表数据: 按天体条目分块
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import logger


class AstronomyContentType(str, Enum):
    STAR_CATALOG = "star_catalog"
    ORBITAL_PARAMETERS = "orbital_parameters"
    ASTROPHYSICS_FORMULA = "astrophysics_formula"
    OBSERVATION_RECORD = "observation_record"
    CELESTIAL_EVENT = "celestial_event"
    INSTRUMENT_SPEC = "instrument_spec"
    SPECTRAL_DATA = "spectral_data"
    GENERAL_TEXT = "general_text"


@dataclass
class AstronomyMetadata:
    doc_type: AstronomyContentType = AstronomyContentType.GENERAL_TEXT
    observation_date: Optional[str] = None
    wavelength_band: Optional[str] = None
    instrument: Optional[str] = None
    celestial_object: Optional[str] = None
    coordinate_system: Optional[str] = None
    data_source: Optional[str] = None
    credibility: float = 1.0
    publication_year: Optional[int] = None
    is_time_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "doc_type": self.doc_type.value,
            "credibility": self.credibility,
            "is_time_sensitive": self.is_time_sensitive,
        }
        if self.observation_date:
            d["observation_date"] = self.observation_date
        if self.wavelength_band:
            d["wavelength_band"] = self.wavelength_band
        if self.instrument:
            d["instrument"] = self.instrument
        if self.celestial_object:
            d["celestial_object"] = self.celestial_object
        if self.coordinate_system:
            d["coordinate_system"] = self.coordinate_system
        if self.data_source:
            d["data_source"] = self.data_source
        if self.publication_year:
            d["publication_year"] = self.publication_year
        return d


_ORBITAL_PARAM_PATTERN = re.compile(
    r'(半长轴|离心率|轨道倾角|升交点经度|近日点幅角|平近点角|'
    r'semi.major|eccentricity|inclination|longitude.*ascending|'
    r'argument.*perihelion|mean.anomaly|'
    r'a\s*[:=]|e\s*[:=]|i\s*[:=]|Ω\s*[:=]|ω\s*[:=]|M\s*[:=])',
    re.IGNORECASE,
)

_FORMULA_PATTERN = re.compile(
    r'(\\frac|\\sum|\\int|\\sqrt|'
    r'σ\s*=|ρ\s*=|'
    r'哈勃|普朗克|斯特藩|维恩|开普勒)',
    re.IGNORECASE,
)

_OBSERVATION_PATTERN = re.compile(
    r'(观测日期|观测时间|观测地点|望远镜|曝光时间|'
    r'observation.date|telescope|exposure|seeing|'
    r'UTC|JD\s|MJD\s|'
    r'视星等|星等|magnitude)',
    re.IGNORECASE,
)

_CATALOG_PATTERN = re.compile(
    r'(HD\s*\d+|HR\s*\d+|HIP\s*\d+|NGC\s*\d+|IC\s*\d+|'
    r'M\s*\d+|Messier|'
    r'赤经|赤纬|RA\s|Dec\s|'
    r'right.ascension|declination|'
    r'光谱型|spectral.type|'
    r'视差|parallax)',
    re.IGNORECASE,
)

_SPECTRAL_PATTERN = re.compile(
    r'(光谱|spectrum|spectral|'
    r'波段|bandpass|filter|'
    r'紫外线|红外线|X射线|射电|'
    r'UV|IR|X.ray|radio|'
    r'波长|wavelength|频率|frequency)',
    re.IGNORECASE,
)

_TIME_SENSITIVE_PATTERN = re.compile(
    r'(新发现|最新|近日|首次|突破|最新观测|'
    r'new.discovery|latest|recent|first|breakthrough|'
    r'20[2-3]\d年)',
    re.IGNORECASE,
)

_WAVELENGTH_BANDS = {
    "radio", "microwave", "infrared", "ir", "visible", "optical",
    "ultraviolet", "uv", "x-ray", "xray", "x射线", "gamma-ray", "gamma", "伽马射线",
    "射电", "微波", "红外", "红外线", "可见光", "紫外", "紫外线", "伽马",
}

_INSTRUMENTS = {
    "hst", "jwst", "alma", "vlt", "keck", "subaru", "gemini",
    "chandra", "xmm", "spitzer", "wise", "gaia", "tess",
    "哈勃", "韦伯", "阿塔卡马", "甚大望远镜", "凯克",
    "钱德拉", "盖亚",
}

_CELESTIAL_OBJECTS = re.compile(
    r'(木星|金星|火星|水星|土星|天王星|海王星|冥王星|'
    r'Jupiter|Venus|Mars|Mercury|Saturn|Uranus|Neptune|Pluto|'
    r'太阳|Sun|月球|Moon|'
    r'仙女座|猎户座|天琴座|天鹅座|大熊座|小熊座|'
    r'M3[1-9]|M4[0-9]|M5[0-9]|M1[0-9]|NGC\s*\d+|IC\s*\d+|'
    r'HD\s*\d+|HIP\s*\d+|'
    r'天狼星|参宿四|织女星|北极星|大角星|'
    r'Sirius|Betelgeuse|Vega|Polaris|Arcturus)',
    re.IGNORECASE,
)


def detect_content_type(text: str) -> AstronomyContentType:
    scores: Dict[AstronomyContentType, int] = {t: 0 for t in AstronomyContentType}

    orbital_matches = len(_ORBITAL_PARAM_PATTERN.findall(text))
    if orbital_matches:
        scores[AstronomyContentType.ORBITAL_PARAMETERS] += orbital_matches

    formula_matches = len(_FORMULA_PATTERN.findall(text))
    if formula_matches:
        scores[AstronomyContentType.ASTROPHYSICS_FORMULA] += formula_matches

    obs_matches = len(_OBSERVATION_PATTERN.findall(text))
    if obs_matches:
        scores[AstronomyContentType.OBSERVATION_RECORD] += obs_matches

    catalog_matches = len(_CATALOG_PATTERN.findall(text))
    if catalog_matches:
        scores[AstronomyContentType.STAR_CATALOG] += catalog_matches

    spectral_matches = len(_SPECTRAL_PATTERN.findall(text))
    if spectral_matches:
        scores[AstronomyContentType.SPECTRAL_DATA] += spectral_matches

    event_kw = re.compile(r'(日食|月食|流星雨|彗星|掩星|冲日|合日|eclipse|meteor|comet|occultation|opposition|conjunction)', re.IGNORECASE)
    if event_kw.search(text):
        scores[AstronomyContentType.CELESTIAL_EVENT] += 3

    inst_kw = re.compile(r'(望远镜|口径|焦比|分辨率|telescope|aperture|focal|resolution)', re.IGNORECASE)
    if inst_kw.search(text):
        scores[AstronomyContentType.INSTRUMENT_SPEC] += 2

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return AstronomyContentType.GENERAL_TEXT
    return best_type


def extract_astronomy_metadata(text: str, base_metadata: Optional[Dict] = None) -> AstronomyMetadata:
    meta = AstronomyMetadata()
    meta.doc_type = detect_content_type(text)

    date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2}|JD\s*[\d.]+|MJD\s*[\d.]+)', text)
    if date_match:
        meta.observation_date = date_match.group(1).strip()

    text_lower = text.lower()
    for band in _WAVELENGTH_BANDS:
        if band in text_lower:
            meta.wavelength_band = band
            break

    for inst in _INSTRUMENTS:
        if inst.lower() in text_lower:
            meta.instrument = inst
            break

    obj_match = _CELESTIAL_OBJECTS.search(text)
    if obj_match:
        meta.celestial_object = obj_match.group(0)

    coord_match = re.search(r'(赤道坐标系|银道坐标系|黄道坐标系|equatorial|galactic|ecliptic)', text, re.IGNORECASE)
    if coord_match:
        meta.coordinate_system = coord_match.group(0)

    if base_metadata:
        meta.data_source = base_metadata.get("source")
        year = base_metadata.get("year") or base_metadata.get("publication_year")
        if year:
            try:
                meta.publication_year = int(year)
            except (ValueError, TypeError):
                pass

    if _TIME_SENSITIVE_PATTERN.search(text):
        meta.is_time_sensitive = True

    year_match = re.search(r'(?:19|20)\d{2}', text)
    if year_match and not meta.publication_year:
        try:
            meta.publication_year = int(year_match.group(0))
        except ValueError:
            pass

    return meta


@dataclass
class ChunkResult:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


class AstronomyChunker:
    """天文领域专用文档分块器"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        preserve_formulas: bool = True,
        preserve_orbital_groups: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_formulas = preserve_formulas
        self.preserve_orbital_groups = preserve_orbital_groups

    def chunk_text(self, text: str, base_metadata: Optional[Dict] = None) -> List[ChunkResult]:
        if not text or not text.strip():
            return []

        content_type = detect_content_type(text)

        if content_type == AstronomyContentType.STAR_CATALOG:
            chunks = self._chunk_catalog(text)
        elif content_type == AstronomyContentType.ORBITAL_PARAMETERS:
            chunks = self._chunk_orbital(text)
        elif content_type == AstronomyContentType.ASTROPHYSICS_FORMULA:
            chunks = self._chunk_formula(text)
        elif content_type == AstronomyContentType.OBSERVATION_RECORD:
            chunks = self._chunk_observation(text)
        elif content_type == AstronomyContentType.CELESTIAL_EVENT:
            chunks = self._chunk_events(text)
        else:
            chunks = self._chunk_general(text)

        results = []
        for i, chunk_content in enumerate(chunks):
            astro_meta = extract_astronomy_metadata(chunk_content, base_metadata)
            merged_meta = {**(base_metadata or {}), **astro_meta.to_dict()}
            results.append(ChunkResult(
                content=chunk_content,
                metadata=merged_meta,
                chunk_index=i,
            ))

        logger.info(
            f"📄 天文分块完成: 类型={content_type.value}, "
            f"原文{len(text)}字 → {len(results)}个分块"
        )
        return results

    def _chunk_catalog(self, text: str) -> List[str]:
        entry_pattern = re.compile(r'(?=(?:HD|HR|HIP|NGC|IC|M)\s*\d+)', re.IGNORECASE)
        parts = entry_pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]

        if not parts:
            return self._chunk_general(text)

        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current = (current + "\n" + part).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_long_text(part))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    def _chunk_orbital(self, text: str) -> List[str]:
        if not self.preserve_orbital_groups:
            return self._chunk_general(text)

        param_group_pattern = re.compile(
            r'((?:半长轴|离心率|轨道倾角|升交点经度|近日点幅角|平近点角|'
            r'semi.major|eccentricity|inclination|longitude|argument|mean.anomaly)'
            r'[^\n]*(?:\n[^\n]*){0,5})',
            re.IGNORECASE,
        )
        groups = param_group_pattern.findall(text)
        if groups:
            remaining = param_group_pattern.sub('', text).strip()
            chunks = [g.strip() for g in groups if g.strip()]
            if remaining:
                chunks.extend(self._chunk_general(remaining))
            return chunks
        return self._chunk_general(text)

    def _chunk_formula(self, text: str) -> List[str]:
        if not self.preserve_formulas:
            return self._chunk_general(text)

        formula_blocks = re.split(r'\n{2,}', text)
        chunks = []
        current = ""
        for block in formula_blocks:
            block = block.strip()
            if not block:
                continue
            if _FORMULA_PATTERN.search(block):
                if current and len(current) + len(block) + 2 <= self.chunk_size:
                    current = current + "\n\n" + block
                else:
                    if current:
                        chunks.append(current)
                    current = block
            else:
                if len(current) + len(block) + 2 <= self.chunk_size:
                    current = (current + "\n\n" + block).strip() if current else block
                else:
                    if current:
                        chunks.append(current)
                    current = block
        if current:
            chunks.append(current)
        return chunks

    def _chunk_observation(self, text: str) -> List[str]:
        record_pattern = re.compile(r'(?=(?:观测日期|观测时间|observation.date|UTC\s+\d))', re.IGNORECASE)
        parts = record_pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]

        if len(parts) <= 1:
            return self._chunk_general(text)

        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current = (current + "\n" + part).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _chunk_events(self, text: str) -> List[str]:
        event_pattern = re.compile(r'(?=(?:日食|月食|流星雨|彗星|掩星|冲日|合日|eclipse|meteor shower|comet|opposition|conjunction))', re.IGNORECASE)
        parts = event_pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]

        if len(parts) <= 1:
            return self._chunk_general(text)

        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) <= self.chunk_size:
                current = (current + "\n" + part).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _chunk_general(self, text: str) -> List[str]:
        return self._split_long_text(text)

    def _split_long_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = re.split(r'\n{2,}', text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= self.chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                    overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current = (overlap_text + "\n\n" + para).strip() if overlap_text else para
                else:
                    sentences = re.split(r'(?<=[。！？.!?])', para)
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) <= self.chunk_size:
                            current += sent
                        else:
                            if current:
                                chunks.append(current)
                            current = sent

        if current:
            chunks.append(current)
        return chunks
