import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from pathlib import Path
from typing import Optional
from config import settings
from logger import logger


# 音频格式映射：文件扩展名 -> DashScope format 参数
_FORMAT_MAP = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".pcm": "pcm",
    ".flac": "flac",
    ".m4a": "m4a",
    ".ogg": "ogg",
    ".aac": "aac",
    ".opus": "opus",
}


class _NoOpCallback(RecognitionCallback):
    """Recognition 构造函数要求 callback 实例，同步 call() 模式下仅 on_error 有意义。"""

    def on_error(self, result: RecognitionResult) -> None:
        logger.error(f"ASR 回调错误: {result}")


class SpeechService:
    """语音识别服务，封装 DashScope ASR Recognition API。"""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.DASHSCOPE_API_KEY
        dashscope.api_key = self._api_key
        self._model_name = settings.SPEECH_MODEL_NAME

    def _detect_format(self, audio_path: str) -> str:
        """根据文件扩展名推断音频格式，默认 wav。"""
        ext = Path(audio_path).suffix.lower()
        return _FORMAT_MAP.get(ext, "wav")

    def transcribe_audio(self, audio_path: str) -> str:
        """将本地音频文件转录为文字。

        Args:
            audio_path: 本地音频文件的绝对路径。

        Returns:
            转录后的文字，失败时返回错误描述字符串。
        """
        try:
            p = Path(audio_path).resolve()

            if not p.exists():
                error_msg = f"音频文件不存在: {audio_path}"
                logger.error(f"❌ {error_msg}")
                return f"语音识别失败：{error_msg}"

            if not p.is_file():
                error_msg = f"路径不是文件: {audio_path}"
                logger.error(f"❌ {error_msg}")
                return f"语音识别失败：{error_msg}"

            audio_format = self._detect_format(audio_path)
            logger.info(f"开始语音识别: {p} (格式={audio_format})")

            # 每次调用需创建新实例，Recognition 内部状态为单次使用
            recognition = Recognition(
                model=self._model_name,
                callback=_NoOpCallback(),
                format=audio_format,
                sample_rate=16000,
            )

            result = recognition.call(file=str(p))

            # 从结果中提取文字
            sentences = result.get_sentence()
            if not sentences:
                logger.warning("语音识别结果为空")
                return "语音内容为空，请重新录制"

            if isinstance(sentences, list):
                texts = [s.get("text", "") for s in sentences if isinstance(s, dict)]
            elif isinstance(sentences, dict):
                texts = [sentences.get("text", "")]
            else:
                texts = [str(sentences)]

            transcription = "".join(texts).strip()
            if not transcription:
                logger.warning("语音识别转录文字为空")
                return "语音内容为空，请重新录制"

            logger.info(f"✅ 语音识别完成: {transcription[:100]}...")
            return transcription

        except Exception as e:
            logger.error(f"❌ 语音识别失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"语音识别失败：{e}"

    def build_speech_query(self, original_query: str, audio_path: str) -> str:
        """将语音转录结果与可选的文本查询组合。

        Args:
            original_query: 用户输入的补充文本（可为空）。
            audio_path: 本地音频文件路径。

        Returns:
            组合后的查询字符串。
        """
        transcription = self.transcribe_audio(audio_path)
        if original_query and original_query.strip():
            return f"{original_query}\n\n[用户语音输入的文字转录]\n{transcription}"
        return transcription
