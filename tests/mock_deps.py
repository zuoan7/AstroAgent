import sys
from unittest.mock import MagicMock


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
        "streamlit", "rich", "langchain_text_splitters",
        "skyfield", "skyfield.api", "skyfield.iokit", "skyfield.timelib",
        "skyfield.toposlib", "skyfield.units", "skyfield.almanac",
        "astropy", "astropy.coordinates", "astropy.time", "astropy.units",
        "astropy.io", "astropy.io.fits",
        "ephem",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

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
