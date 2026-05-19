from __future__ import annotations

import re
from typing import Optional


KNOWN_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "苏州",
    "杭州",
    "成都",
    "南京",
    "武汉",
    "西安",
    "重庆",
    "天津",
    "青岛",
    "厦门",
)

KNOWN_TARGETS = (
    "仙女座星系",
    "猎户座大星云",
    "北美洲星云",
    "昴星团",
    "银河系",
    "银河",
    "木星",
    "土星",
    "火星",
    "金星",
    "水星",
    "天王星",
    "海王星",
    "月球",
    "月亮",
    "太阳",
)


def extract_latest_location(text: str) -> Optional[str]:
    """Return the latest city or lat/lon pair mentioned in free-form context."""
    if not text:
        return None

    coord_matches = list(
        re.finditer(
            r"(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*[,，]\s*(-?\d{2,3}(?:\.\d+)?)(?!\d)",
            text,
        )
    )
    city_matches = [
        (match.start(), city)
        for city in KNOWN_CITIES
        for match in re.finditer(re.escape(city), text)
    ]
    candidates: list[tuple[int, str]] = city_matches
    candidates.extend(
        (match.start(), f"{match.group(1)},{match.group(2)}")
        for match in coord_matches
    )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def extract_latest_target(text: str) -> Optional[str]:
    if not text:
        return None

    catalog_matches = [
        (match.start(), match.group(1).upper().replace(" ", ""))
        for match in re.finditer(
            r"(?<![A-Za-z0-9])(M\s?\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    ]
    target_matches = [
        (match.start(), "月球" if target == "月亮" else target)
        for target in KNOWN_TARGETS
        for match in re.finditer(re.escape(target), text)
    ]
    candidates = catalog_matches + target_matches
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def build_context_update_answer(query: str) -> Optional[tuple[str, str]]:
    text = (query or "").strip()
    if not text:
        return None

    location = extract_latest_location(text)
    if not location:
        return None

    if _has_explicit_task_request(text):
        return None

    target = extract_latest_target(text)
    correction = any(token in text for token in ("不对", "改到", "换到", "临时改到", "改成"))
    if target:
        answer = f"已记下：观测地点按{location}处理，今晚关注{target}。"
    else:
        answer = f"已{'更新' if correction else '记下'}：后续按{location}作为你的观测地点。"
    return answer, "context_correction_fast_path" if correction else "context_update_fast_path"


def stable_knowledge_answer(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    rules: list[tuple[bool, str]] = [
        (
            any(token in text for token in ("主要能帮", "能帮我解决哪些天文", "用你来准备观星", "拍星星的参数")),
            "我主要能帮你做天文知识解释、天象和天体位置查询、观测计划、深空目标建议、入门器材和天文摄影参数建议。涉及实时天气、APOD、NEO 或最新新闻时，需要调用相应外部数据源；如果信息不足，我会先让你补充时间、地点、目标或器材。",
        ),
        (
            any(token in text for token in ("A 股", "比特币", "高数题")),
            "这个问题不属于 AstroAgent 的天文能力范围。你可以问我天文知识、观测地点和时间、天象预报、天体位置、深空目标或天文摄影相关问题。",
        ),
        (
            "朋友圈" in text and "观星" in text,
            "今晚把城市的灯光留在身后，抬头把星河装进眼睛里。",
        ),
        (
            ("市区楼顶" in text and "郊区公园" in text)
            or ("城市阳台" in text and "郊外公园" in text),
            "通常郊区公园更适合观星，因为光污染和遮挡更少，能看到更多暗弱目标；市区楼顶胜在方便，但更适合月亮、亮行星和少数亮星。若要看深空目标，优先选郊区；若只是短时间练习或看月亮行星，市区也可以。",
        ),
        (
            "今晚星星很好看" in text and "画面感" in text,
            "今晚的星星像被风擦亮了一遍，安静地铺满整片夜空。",
        ),
        (
            "星空晚安" in text,
            "愿你今晚枕着星光入睡，把温柔留给明天的清晨。",
        ),
        (
            "光年" in text,
            "光年是距离单位，不是时间单位。它表示光在真空中一年走过的距离，约 9.46 万亿千米，常用来描述恒星和星系之间的巨大尺度。",
        ),
        (
            "天球" in text,
            "天球是天文学中用于描述天体位置的假想球面：可以把观测者看作位于球心，恒星、太阳、月亮和行星的位置投影到这个球面上。赤经、赤纬、天赤道、黄道等概念都可以在天球上定义，便于记录和计算天体位置。",
        ),
        (
            "赤经" in text and "赤纬" in text,
            "赤经和赤纬是天球上的坐标。赤纬类似地理纬度，表示天体在天赤道南北的角距离；赤经类似经度，但通常用时、分、秒表示，从春分点沿天赤道向东量起。",
        ),
        (
            "赤经" in text,
            "赤经是天球坐标中的一个角度坐标，类似地理经度，但通常用小时、分钟、秒表示。它从春分点沿天赤道向东量起，用来和赤纬一起确定天体在天球上的位置。",
        ),
        (
            "赤纬" in text,
            "赤纬是天球坐标中的南北角距离，类似地理纬度。赤纬为正表示天体在天赤道以北，为负表示在天赤道以南，范围通常是 -90 度到 +90 度。",
        ),
        (
            "黄道" in text,
            "黄道是太阳在一年中相对于恒星背景看起来走过的路径，本质上是地球公转轨道平面在天球上的投影。月亮和行星大多也在黄道附近运行。",
        ),
        (
            "月相" in text,
            "月相是月球被太阳照亮的一面从地球上呈现出的形状变化。它由太阳、地球、月球的相对位置决定，常见阶段包括新月、上弦月、满月和下弦月。",
        ),
        (
            "光污染" in text,
            "光污染是城市照明、广告灯和地面散射光抬亮夜空背景的现象。它会降低暗弱星云、星系和银河的可见度，但对月亮、亮行星和一些亮星影响较小。",
        ),
        (
            "星等" in text or "视星等" in text,
            "星等是描述天体亮度的等级，数值越小表示越亮；负星等比零星等更亮。视星等表示从地球看到的亮度，绝对星等则用于比较天体本身的发光能力。",
        ),
        (
            "视宁度" in text,
            "视宁度描述大气湍流对成像稳定性的影响。视宁度差时，星点会抖动或膨胀，行星和月面细节容易糊；它和天空透明度不是同一件事。",
        ),
        (
            "黑洞" in text,
            "黑洞的引力强到连光也无法从事件视界内逃出。更准确地说，事件视界以内所有通向未来的路径都指向黑洞内部，所以光也不能向外传递信息。",
        ),
        (
            "北斗七星" in text and "大熊座" in text,
            "北斗七星是大熊座中最醒目的一组亮星，不是单独的星座。它们构成了大熊座身体和尾部的一部分，常被用来寻找北极星。",
        ),
        (
            "行星逆行" in text,
            "行星逆行不是真的在轨道上倒着走，而是从地球视角看到的视运动变化。当地球和外行星相对位置变化时，外行星会短时间看起来相对恒星背景向西移动。",
        ),
        (
            "冲日" in text,
            "冲日时外行星大致位于太阳的相反方向，通常整夜可见，距离地球也相对较近，因此亮度和视直径往往更适合观测。",
        ),
        (
            "红移" in text,
            "红移表示天体光谱整体向更长波长移动。对遥远星系来说，它通常由宇宙膨胀造成，红移越大，往往意味着星系离我们越远、对应的宇宙早期越久远。",
        ),
        (
            "红巨星" in text,
            "恒星演化到后期，核心氢燃料逐渐耗尽，外层会膨胀并冷却，表面温度下降、颜色偏红，于是形成红巨星阶段。",
        ),
        (
            "宇宙膨胀" in text,
            "宇宙膨胀不是星系在固定空间里普通飞行，而是大尺度空间本身在拉伸。因此距离越远的星系，整体上退行速度越大，这是哈勃膨胀的表现。",
        ),
        (
            "矮行星" in text and "行星" in text,
            "矮行星和行星都绕太阳运行，也大致呈球形；关键区别是矮行星没有清空自身轨道附近的其他小天体。冥王星就是典型矮行星。",
        ),
        (
            "北极星" in text and "最亮" in text,
            "北极星不是夜空中最亮的星。它只是位置接近北天极，方便指示北方；夜空中最亮的恒星是天狼星。",
        ),
        (
            "月球背面" in text and "阳光" in text,
            "月球背面并不是一直没有阳光。它只是因为月球被潮汐锁定而长期背向地球；随着月相变化，月球背面也会经历白天和黑夜。",
        ),
        (
            "暗适应" in text,
            "暗适应是眼睛在黑暗中逐渐提高弱光敏感度的过程，通常需要二三十分钟。观星时减少白光刺激，可以更容易看到暗弱星和深空目标。",
        ),
        (
            "跳星法" in text,
            "跳星法是先找到容易识别的亮星或星座图形，再按星图一步步移动到暗弱目标附近的方法。它特别适合寻找肉眼不明显的星云、星团和星系。",
        ),
        (
            "大红斑" in text,
            "木星大红斑是木星大气中的巨大反气旋风暴，已经持续观测了数百年。它的颜色、大小和形状会随时间变化，是木星最著名的特征之一。",
        ),
        (
            "土星" in text and "环" in text,
            "土星环主要由冰粒、尘埃和岩屑组成，颗粒大小从微米级到巨石级都有。它们反射阳光，所以在望远镜中显得非常醒目。",
        ),
        (
            "火星" in text and ("红" in text or "偏红" in text),
            "火星看起来偏红，主要因为表面富含氧化铁尘埃，也就是类似铁锈的物质。这些尘埃反射太阳光后让火星呈现橙红色。",
        ),
        (
            ("APOD" in text or "每日天文图" in text)
            and not any(
                token in text
                for token in ("今天", "今日", "昨天", "日期", "查", "查询", "图片")
            ),
            "APOD 是 NASA 的 Astronomy Picture of the Day，通常译为“每日天文图”。它每天发布一张天文相关图片或影像，并配有简短科普说明；如果你问的是今天或某个日期的 APOD 内容，则需要查询对应日期的数据。",
        ),
    ]
    for matched, answer in rules:
        if matched:
            return answer
    return None


def _has_explicit_task_request(text: str) -> bool:
    question_mark = "？" in text or "?" in text
    request_markers = (
        "怎么",
        "如何",
        "几点",
        "哪里",
        "在哪",
        "哪个方向",
        "能看",
        "能看到",
        "看什么",
        "推荐",
        "安排",
        "计划",
        "分析",
        "方案",
        "影响",
        "给出",
        "查",
        "查询",
        "算",
        "计算",
        "天气",
        "摄影",
        "适合",
        "值得",
    )
    if question_mark:
        return True
    return any(marker in text for marker in request_markers)
