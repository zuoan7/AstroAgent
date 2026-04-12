import dashscope
from dashscope import MultiModalConversation
from pathlib import Path
from typing import Optional
from src.core.config import settings
from src.core.logger import logger
from src.core.errors import AgentError, ErrorCode, ErrorHandler


class VisionService:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.DASHSCOPE_API_KEY
        dashscope.api_key = self._api_key

    def describe_image(self, image_path: str, prompt: str) -> str:
        if not self._api_key:
            raise AgentError(
                code=ErrorCode.VISION_ERROR,
                message="DASHSCOPE_API_KEY 未配置，无法使用视觉服务",
            )
        try:
            logger.info(f"尝试读取图片: {image_path}")
            p = Path(image_path).resolve()

            if not p.exists():
                raise AgentError(
                    code=ErrorCode.FILE_NOT_FOUND,
                    message=f"图片文件不存在: {image_path}",
                    details={"image_path": image_path}
                )

            if not p.is_file():
                raise AgentError(
                    code=ErrorCode.FILE_NOT_FOUND,
                    message=f"路径不是文件: {image_path}",
                    details={"image_path": image_path}
                )

            logger.info(f"图片文件存在: {p}")
            image_uri = f"file://{p}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_uri},
                        {"text": prompt},
                    ],
                }
            ]
            resp = MultiModalConversation.call(
                model=settings.VISION_MODEL_NAME,
                messages=messages,
            )
            if isinstance(resp, dict):
                out = resp.get("output") or {}
                choices = out.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(str(item["text"]))
                        return "\n".join(texts).strip()
            return str(resp)
        except AgentError:
            raise
        except Exception as e:
            logger.error(f"❌ 图片理解失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise AgentError(
                code=ErrorCode.VISION_ERROR,
                message=f"图片理解失败: {e}",
                details={"image_path": image_path},
                original_error=e
            )

    def build_vision_query(self, original_query: str, image_path: str, custom_prompt: Optional[str] = None) -> str:
        prompt = custom_prompt or (
            "请详细描述这张图片的天文相关内容。若图片包含星空/天体/天文设备，请按以下结构分析：\n"
            "1) 天体识别：指出可能的天体名称、星座、天文现象（如流星、极光、日/月食等），"
            "尽量给出中文名和编号（如猎户座大星云 M42）\n"
            "2) 天空质量评估：根据可见星点数量和暗弱程度，判断光害等级（Bortle暗空等级）和天空透明度\n"
            "3) 设备与拍摄参数推断：若可见望远镜/赤道仪/相机，描述设备类型；"
            "若为天文照片，尝试推断曝光时间、ISO、焦距等参数\n"
            "4) 观测与拍摄建议：基于上述分析，给出适合的后续观测目标、拍摄参数或设备升级建议"
        )
        try:
            vision_desc = self.describe_image(image_path=image_path, prompt=prompt)
        except AgentError as e:
            vision_desc = f"图片理解失败: {e.message}"
        return f"{original_query}\n\n[用户上传图片的视觉信息]\n{vision_desc}"
