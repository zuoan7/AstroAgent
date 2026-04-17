import sys
import importlib.util
from unittest.mock import MagicMock


def _is_module_available(mod_name):
    if mod_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


def _mock_tenacity():
    class _IdentityDecorator:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, fn):
            return fn

    mock = MagicMock()
    mock.retry = _IdentityDecorator
    mock.stop_after_attempt = MagicMock(return_value=None)
    mock.wait_exponential = MagicMock(return_value=None)
    mock.retry_if_exception_type = MagicMock(return_value=(Exception,))
    mock.before_sleep = MagicMock(return_value=None)
    return mock


def _mock_pybreaker():
    class _MockCircuitBreaker:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, fn):
            return fn

    mock = MagicMock()
    mock.CircuitBreaker = _MockCircuitBreaker
    return mock


def mock_heavy_dependencies():
    for mod_name in [
        "langchain", "langchain.schema",
        "langchain_community", "langchain_community.chat_models",
        "langchain_community.embeddings",
        "langchain_core", "langchain_core.prompts", "langchain_core.tools",
        "langchain_core.callbacks", "langchain_core.messages",
        "langchain_classic", "langchain_classic.agents",
        "langchain_chroma", "langgraph", "dashscope",
        "dashscope.audio", "dashscope.audio.asr",
        "dashscope.multi_modal", "dashscope.protocol",
        "chromadb", "rank_bm25", "fastmcp", "langchain_mcp_adapters",
        "astroquery", "astroquery.simbad", "astroquery.ned",
        "rich", "langchain_text_splitters",
        "skyfield", "skyfield.api", "skyfield.iokit", "skyfield.timelib",
        "skyfield.toposlib", "skyfield.units", "skyfield.almanac",
        "astropy", "astropy.coordinates", "astropy.time", "astropy.units",
        "astropy.io", "astropy.io.fits",
        "ephem",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    if not _is_module_available("tenacity"):
        sys.modules["tenacity"] = _mock_tenacity()

    if not _is_module_available("pybreaker"):
        sys.modules["pybreaker"] = _mock_pybreaker()

    import src.core.config
    import src.core.errors
    import src.core.logger
    import src.agent.param_parser

    for mod_name in [
        "src.agent.fallback_service",
        "src.agent.speech_service",
        "src.agent.streaming_service",
        "src.agent.vision_service",
        "src.agent.skill_manager",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
