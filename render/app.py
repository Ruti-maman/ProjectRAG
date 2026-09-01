"""
RAG Agent - Render entry point
==============================
Wraps RAG_Agent_Project/app_local.py for a public Linux host.
Nothing under RAG_Agent_Project/ is modified.

No SSL bypass here: verification is disabled in app_local.py to survive an
office TLS-intercepting proxy, and turning it off on a public host would hand
anyone on the path the Cohere API key.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Gradio phones home for a version check on startup; on a cold-starting free
# instance that is dead time before the port is bound.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

PROJECT_DIR = Path(__file__).resolve().parent.parent / "RAG_Agent_Project"
sys.path.insert(0, str(PROJECT_DIR))
# Her code addresses ./storage_local and ./dummy_data relatively, so the process
# has to live where she expects it to.
os.chdir(PROJECT_DIR)

import gradio as gr  # noqa: E402
from llama_index.core import (  # noqa: E402
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.prompts import PromptTemplate  # noqa: E402
from llama_index.embeddings.cohere import CohereEmbedding  # noqa: E402
from llama_index.llms.cohere import Cohere  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("render.app")

STORAGE_DIR = PROJECT_DIR / "storage_local"
DOCS_DIR = PROJECT_DIR / "dummy_data"

EMBED_MODEL = "embed-multilingual-v3.0"
LLM_MODEL = "command-r-plus-08-2024"


def require_api_key() -> str:
    key = os.getenv("COHERE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "\n"
            "COHERE_API_KEY is not set.\n"
            "On Render: Dashboard -> your service -> Environment -> Add Environment Variable\n"
            "  Key:   COHERE_API_KEY\n"
            "  Value: your key from https://dashboard.cohere.com/api-keys\n"
            "Then redeploy. The app cannot embed or answer anything without it.\n"
        )
    return key


# ========================================
# RAG System
# ========================================
# Copied rather than imported: app_local.py runs its SSL-bypass block at module
# level, so `import app_local` would reinstall exactly the thing this file
# exists to avoid (and it raises on a missing key before any of it is usable).
# Same for ingest_local.py, hence build_index() below.
def setup_llama_index(api_key: str, input_type: str = "search_query") -> None:
    Settings.embed_model = CohereEmbedding(
        api_key=api_key,
        model_name=EMBED_MODEL,
        input_type=input_type,
    )
    Settings.llm = Cohere(api_key=api_key, model=LLM_MODEL, temperature=0.7)
    logger.info("Cohere embeddings + LLM configured")


def build_index(api_key: str) -> None:
    """Mirrors ingest_local.py: read dummy_data/, chunk, embed, persist."""
    logger.info("Building index: %s -> %s", DOCS_DIR, STORAGE_DIR)

    Settings.embed_model = CohereEmbedding(
        api_key=api_key,
        model_name=EMBED_MODEL,
        input_type="search_document",
    )

    documents = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    logger.info("Loaded %d documents", len(documents))

    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
    logger.info("Split into %d chunks", len(nodes))

    index = VectorStoreIndex(nodes, show_progress=False)
    STORAGE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    logger.info("Index persisted to %s", STORAGE_DIR)


class RAGSystem:
    def __init__(self, api_key: str):
        setup_llama_index(api_key)
        self.index = self._load_index()

        qa_prompt = PromptTemplate(
            """אתה עוזר AI מומחה ב-RAG (Retrieval-Augmented Generation).

ענה על השאלה בהתבסס על המסמכים הבאים:
{context_str}

שאלה: {query_str}

תשובה (בעברית, תמציתית ומדויקת):"""
        )

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            text_qa_template=qa_prompt,
        )
        logger.info("RAG System ready")

    def _load_index(self):
        logger.info("Loading index from %s", STORAGE_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(storage_context)

    def query(self, question: str) -> str:
        if not question.strip():
            return "❓ נא להזין שאלה"
        try:
            return str(self.query_engine.query(question))
        except Exception as e:
            logger.error("Query error: %s", e)
            return f"⚠️ שגיאה: {e}"


# ========================================
# Gradio UI
# ========================================
def create_gradio_app(rag_system: RAGSystem):
    custom_css = """
    .gradio-container {
        max-width: 620px !important;
        margin: 50px auto !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(180, 150, 180, 0.2) !important;
        background: #fefefe !important;
        overflow: hidden !important;
    }
    .main-header {
        background: linear-gradient(135deg, #e8d5e8 0%, #d4c4e8 50%, #c8d4e8 100%);
        padding: 26px 20px;
        text-align: center;
    }
    .main-header h1 {
        color: #5a4a6a;
        font-size: 1.9em;
        margin: 0 0 6px 0;
        font-weight: 600;
    }
    .main-header p {
        color: #6a5a7a;
        font-size: 0.92em;
        margin: 0 0 14px 0;
    }
    .badges {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.6);
        padding: 5px 12px;
        border-radius: 14px;
        color: #5a4a6a;
        font-size: 0.76em;
        font-weight: 500;
    }
    .free-tier-note {
        background: #fdf6e3;
        border-right: 4px solid #e0c98f;
        color: #6a5a3a;
        padding: 10px 14px;
        margin: 12px 16px 0 16px;
        border-radius: 10px;
        font-size: 0.82em;
        line-height: 1.55;
        direction: rtl;
        text-align: right;
    }
    """

    header_html = """
    <div class="main-header">
        <h1>🤖 RAG Agent</h1>
        <p>חיפוש סמנטי חכם בתיעוד פרויקט</p>
        <div class="badges">
            <span class="badge">🔮 Cohere</span>
            <span class="badge">💾 Local</span>
            <span class="badge">⚡ LlamaIndex</span>
        </div>
    </div>
    """

    # A visitor who hits a cold instance sees a blank tab for the better part of
    # a minute; without this they assume it is broken and leave.
    free_tier_html = """
    <div class="free-tier-note">
        ⏳ <b>הערה:</b> האתר מתארח בשכבה החינמית של Render, ולכן הוא נכנס למצב שינה
        לאחר 15 דקות ללא שימוש. אם הטעינה הראשונה איטית — זה תקין. המתינו כ-50 שניות
        עד שהשרת מתעורר, ומשם התשובות מגיעות תוך שניות בודדות.
    </div>
    """

    def chat_fn(message, history):
        if not message.strip():
            return ""
        return rag_system.query(message)

    with gr.Blocks(title="RAG Agent") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.HTML(header_html)
        gr.HTML(free_tier_html)

        gr.ChatInterface(
            fn=chat_fn,
            examples=["מה זה RAG?", "למה RAG חשוב?", "איך RAG עובד?"],
        )

    return demo


# ========================================
# Main
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build storage_local and exit. Run from buildCommand so the index is "
             "baked into the deploy and cold starts skip the embedding round-trip.",
    )
    args = parser.parse_args()

    if args.build_index:
        # A missing key at build time is not worth failing the whole deploy over;
        # boot will build the index once the key is set in the dashboard.
        key = os.getenv("COHERE_API_KEY", "").strip()
        if not key:
            logger.warning("COHERE_API_KEY absent at build time - deferring ingest to first boot")
            return
        build_index(key)
        return

    api_key = require_api_key()

    if not STORAGE_DIR.exists():
        logger.info("No index at %s - building it from %s", STORAGE_DIR, DOCS_DIR)
        build_index(api_key)
    else:
        logger.info("Existing index found at %s", STORAGE_DIR)

    rag_system = RAGSystem(api_key)

    port = int(os.environ.get("PORT", 7860))
    logger.info("Launching Gradio on 0.0.0.0:%d", port)
    demo = create_gradio_app(rag_system)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
