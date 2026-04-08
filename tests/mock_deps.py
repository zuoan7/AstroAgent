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
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
