import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from pathlib import Path
from typing import Optional
from config import settings
from logger import logger
from core.errors import AgentError, ErrorCode


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

    def on_error(self, result: RecognitionResult) -> None:
        logger.error(f"ASR 回调错误: {result}")


class SpeechService:

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.DASHSCOPE_API_KEY
        dashscope.api_key = self._api_key
        self._model_name = settings.SPEECH_MODEL_NAME

    def _detect_format(self, audio_path: str) -> str:
        ext = Path(audio_path).suffix.lower()
        return _FORMAT_MAP.get(ext, "wav")

    def transcribe_audio(self, audio_path: str) -> str:
        try:
            p = Path(audio_path).resolve()

            if not p.exists():
                raise AgentError(
                    code=ErrorCode.FILE_NOT_FOUND,
                    message=f"音频文件不存在: {audio_path}",
                    details={"audio_path": audio_path}
                )

            if not p.is_file():
                raise AgentError(
                    code=ErrorCode.FILE_NOT_FOUND,
                    message=f"路径不是文件: {audio_path}",
                    details={"audio_path": audio_path}
                )

            audio_format = self._detect_format(audio_path)
            logger.info(f"开始语音识别: {p} (格式={audio_format})")

            recognition = Recognition(
                model=self._model_name,
                callback=_NoOpCallback(),
                format=audio_format,
                sample_rate=16000,
            )

            result = recognition.call(file=str(p))

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

        except AgentError:
            raise
        except Exception as e:
            logger.error(f"❌ 语音识别失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise AgentError(
                code=ErrorCode.SPEECH_ERROR,
                message=f"语音识别失败: {e}",
                details={"audio_path": audio_path},
                original_error=e
            )

    def build_speech_query(self, original_query: str, audio_path: str) -> str:
        try:
            transcription = self.transcribe_audio(audio_path)
        except AgentError as e:
            transcription = f"语音识别失败: {e.message}"
        if original_query and original_query.strip():
            return f"{original_query}\n\n[用户语音输入的文字转录]\n{transcription}"
        return transcription
