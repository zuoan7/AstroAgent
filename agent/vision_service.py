import dashscope
from dashscope import MultiModalConversation
from pathlib import Path
from typing import Optional
from config import settings
from logger import logger


class VisionService:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.DASHSCOPE_API_KEY
        dashscope.api_key = self._api_key

    def describe_image(self, image_path: str, prompt: str) -> str:
        try:
            logger.info(f"尝试读取图片: {image_path}")
            p = Path(image_path).resolve()
            
            if not p.exists():
                error_msg = f"图片文件不存在: {image_path}"
                logger.error(f"❌ {error_msg}")
                return f"图片读取失败：{error_msg}"
            
            if not p.is_file():
                error_msg = f"路径不是文件: {image_path}"
                logger.error(f"❌ {error_msg}")
                return f"图片读取失败：{error_msg}"
            
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
        except Exception as e:
            logger.error(f"❌ 图片理解失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"图片理解失败：{e}"

    def build_vision_query(self, original_query: str, image_path: str, custom_prompt: Optional[str] = None) -> str:
        prompt = custom_prompt or (
            "请详细描述这张图片的内容。若包含星空/天体/望远镜设备，请指出："
            "1) 可能的天体/星座/现象；2) 光害/天空质量线索；3) 设备与拍摄参数线索；"
            "4) 适合的后续观测或拍摄建议。"
        )
        vision_desc = self.describe_image(image_path=image_path, prompt=prompt)
        return f"{original_query}\n\n[用户上传图片的视觉信息]\n{vision_desc}"
