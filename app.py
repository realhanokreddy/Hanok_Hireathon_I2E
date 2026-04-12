"""
NASA Technical Manual QA System - Streamlit UI
Clean chat interface for querying the NASA Systems Engineering Handbook.
"""
import streamlit as st
import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
env_path = Path(__file__).parent / 'env' / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.generation.qa_system import TechnicalQASystem
from src.config import get_config

# Load logo for page icon and header
_LOGO_PATH = Path(__file__).parent / "logo.png"
try:
    from PIL import Image as _PILImage
    _page_icon = _PILImage.open(_LOGO_PATH)
except Exception:
    _page_icon = "🚀"

try:
    _logo_b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    _LOGO_IMG_TAG = f'<img src="data:image/png;base64,{_logo_b64}" style="height:64px;vertical-align:middle;margin-right:12px;">'
except Exception:
    _LOGO_IMG_TAG = ""

# Page configuration
st.set_page_config(
    page_title="CLINICAL RESEARCH OPERATIONS MANUAL",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ───────────────────────────────────────────── */
    .stApp {
        background: #ffffff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Header ────────────────────────────────────────────────────── */
    .main-header {
        display: flex;
        align-items: center;
        background: linear-gradient(135deg, #0b3d91 0%, #1260d4 100%);
        border-radius: 16px;
        padding: 1rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(11,61,145,0.18);
    }

    .main-header-text {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    .main-header-sub {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.72);
        margin-top: 2px;
        letter-spacing: 0.4px;
    }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #f4f7fc !important;
        border-right: 1px solid #dde6f5 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0b3d91 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid #c5d6f0 !important;
        color: #1a3a7c !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #0b3d91 !important;
        border-color: #0b3d91 !important;
        color: #ffffff !important;
        transform: translateX(3px);
    }

    /* ── Welcome card ──────────────────────────────────────────────── */
    .welcome-card {
        background: linear-gradient(135deg, #eef4ff 0%, #f5f8ff 100%);
        border: 1px solid #c5d6f0;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 2px 12px rgba(11,61,145,0.07);
    }

    .welcome-card h2 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0b3d91;
        margin-bottom: .5rem;
    }

    .welcome-card p {
        color: #4a6080;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* ── Chat messages ─────────────────────────────────────────────── */
    .user-message {
        background: linear-gradient(135deg, #0b3d91 0%, #1260d4 100%);
        color: #ffffff;
        padding: 1rem 1.4rem;
        border-radius: 20px 20px 4px 20px;
        margin: 0.8rem 0;
        margin-left: 15%;
        box-shadow: 0 4px 16px rgba(11,61,145,0.2);
        line-height: 1.6;
    }

    .user-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .assistant-message {
        background: #ffffff;
        color: #1a2a40;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 20px 4px;
        margin: 0.8rem 0;
        margin-right: 10%;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border: 1px solid #dde6f5;
        border-left: 4px solid #0b3d91;
        line-height: 1.7;
    }

    .assistant-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        color: #0b3d91;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ── Citation badges ──────────────────────────────────────────── */
    .citation-badge {
        display: inline-block;
        background: #eef4ff;
        border: 1px solid #c0d4f5;
        color: #1260d4;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 0.25rem 0.2rem;
        font-weight: 500;
        letter-spacing: 0.2px;
        cursor: default;
        transition: background 0.2s;
    }

    .citation-badge:hover {
        background: #d6e6ff;
    }

    /* ── Metadata bar ─────────────────────────────────────────────── */
    .message-metadata {
        font-size: 0.75rem;
        color: #8a9bb8;
        margin-top: 0.75rem;
        padding-top: 0.6rem;
        border-top: 1px solid #e8eef8;
        letter-spacing: 0.3px;
    }

    /* ── Sources label ────────────────────────────────────────────── */
    .sources-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #5a78a8;
        text-transform: uppercase;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    /* ── Input area ───────────────────────────────────────────────── */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1.5px solid #c5d6f0 !important;
        border-radius: 14px !important;
        color: #1a2a40 !important;
        font-size: 0.97rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextArea textarea:focus {
        border-color: #1260d4 !important;
        box-shadow: 0 0 0 3px rgba(18,96,212,0.1) !important;
    }

    /* ── Send button ──────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0b3d91 0%, #1260d4 100%) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.25s !important;
        box-shadow: 0 4px 14px rgba(11,61,145,0.3) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1260d4 0%, #1a78ff 100%) !important;
        box-shadow: 0 6px 20px rgba(11,61,145,0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Stat card ────────────────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(135deg, #eef4ff 0%, #e0ecff 100%);
        border: 1px solid #c5d6f0;
        color: #1a3a7c;
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0b3d91;
    }

    .stat-label {
        font-size: 0.82rem;
        color: #5a78a8;
        letter-spacing: 0.5px;
    }

    /* ── Dividers & misc ──────────────────────────────────────────── */
    hr {
        border-color: #dde6f5 !important;
    }

    [data-testid="stStatusWidget"] { display: none; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f0f4fb; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #c0d0e8; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #90aad0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_qa_system():
    """Load the QA system (cached for performance)."""
    try:
        # Check if vector store exists
        config = get_config()
        vector_store_path = config.get('paths.vector_store')
        
        if not Path(vector_store_path).exists() or not any(Path(vector_store_path).iterdir()):
            return None, "Vector store not found. Run: python build_vectorstore.py"
        
        qa_system = TechnicalQASystem()
        return qa_system, None
    except Exception as e:
        return None, str(e)


def main():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.system_error = None
    
    # Header
    st.markdown(
        f'''
        <div class="main-header">
            {_LOGO_IMG_TAG}
            <div>
                <div class="main-header-text">CLINICAL RESEARCH OPERATIONS MANUAL</div>
                <div class="main-header-sub">AI-Powered Technical Assistant &nbsp;·&nbsp; Intelligent Q&amp;A with Verified Citations</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Load QA system (only once)
        if st.session_state.qa_system is None:
            with st.spinner("🔄 Loading QA system..."):
                qa_system, error = load_qa_system()
                st.session_state.qa_system = qa_system
                st.session_state.system_error = error
        
        # System status
        if st.session_state.system_error:
            st.error(f"❌ {st.session_state.system_error}")
            st.info("💡 Run: `python build_vectorstore.py`")
            st.stop()
        else:
            st.success("✅ System Ready")
            if st.session_state.qa_system:
                n = len(st.session_state.qa_system.vector_store.chunks)
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#5a78a8;margin-top:-0.3rem;">'
                    f'📚 {n:,} knowledge chunks indexed</div>',
                    unsafe_allow_html=True,
                )
        
        st.markdown("---")
        
        # Example queries
        st.subheader("💡 Try These Questions")
        examples = [
            "What is TRL?",
            "Key phases of SE?",
            "Verification vs validation?",
            "PDR entry criteria?",
            "Risk management process?"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": example
                })
                # Trigger rerun to process
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Advanced"):
            show_metadata = st.checkbox("Show metadata", value=True)
            show_citations = st.checkbox("Show citations", value=True)
        
        # Stats
        if st.session_state.messages:
            st.markdown("---")
            st.subheader("📊 Session Stats")
            
            num_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f'<div class="stat-card"><div class="stat-value">{num_questions}</div><div class="stat-label">Questions Asked</div></div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat area
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        if not st.session_state.messages:
            st.markdown(
                '''
                <div class="welcome-card">
                    <h2>👋 Welcome to the Clinical Research Operations Manual Assistant</h2>
                    <p>
                        Ask any question about the <strong>Clinical Research Operations Manual</strong>.<br>
                        Answers are grounded in the source document with verified citations.
                    </p>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="user-message">'
                        f'<div class="user-label">You</div>'
                        f'{message["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Assistant message
                    answer_html = '<div class="assistant-message">'
                    answer_html += '<div class="assistant-label">🛰 Clinical Research Operations Manual Assistant</div>'
                    answer_html += message["content"]
                    
                    # Add metadata
                    if show_metadata and "metadata" in message:
                        meta = message["metadata"]
                        answer_html += '<div class="message-metadata">'
                        answer_html += f'⚡ {meta["time"]:.2f}s | '
                        answer_html += f'📊 Confidence: {meta["confidence"]:.0%} | '
                        answer_html += f'📄 {meta["chunks"]} chunks'
                        if meta.get("multi_hop"):
                            answer_html += ' | 🔗 Multi-hop'
                        answer_html += '</div>'
                    
                    # Add citations
                    if show_citations and "citations" in message and message["citations"]:
                        # Build a lookup: section_number -> verification status
                        verif_map = {}
                        for v in message.get("citation_verification", []):
                            verif_map[v["section_number"]] = v

                        STATUS_ICON = {
                            "verified": "✅",
                            "partial":  "⚠️",
                            "not_found": "❌",
                        }
                        STATUS_TITLE = {
                            "verified":  "Citation verified — content found in the cited section",
                            "partial":   "Partial match — section exists but content overlap is low or page may differ",
                            "not_found": "Citation not found — no chunk with this section number",
                        }

                        answer_html += '<div class="sources-label">📎 Sources</div>'
                        for cit in message["citations"][:5]:  # Show top 5
                            sec = cit["section_number"]
                            v = verif_map.get(sec)
                            if v:
                                icon  = STATUS_ICON.get(v["status"], "")
                                title = STATUS_TITLE.get(v["status"], "")
                                score = f' ({v["grounding_score"]:.0%})' if v["grounding_score"] > 0 else ""
                                answer_html += (
                                    f'<span class="citation-badge" title="{title}">'
                                    f'{icon} Section {sec} (p.{cit["page_start"]}){score}</span>'
                                )
                            else:
                                answer_html += f'<span class="citation-badge">Section {sec} (p.{cit["page_start"]})</span>'
                        answer_html += '</div>'
                    
                    answer_html += '</div>'
                    st.markdown(answer_html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Input area at the bottom
    st.markdown('<hr style="margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Ask your question:",
            placeholder="e.g., What are the key activities in systems engineering?",
            height=100,
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("🚀 Send", use_container_width=True, type="primary")
    
    # Process new message
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                result = st.session_state.qa_system.ask(user_input.strip(), include_context=False)
                elapsed = time.time() - start_time
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                    "citation_verification": result.get("citation_verification", []),
                    "metadata": {
                        "time": elapsed,
                        "confidence": result.get("confidence", 0),
                        "chunks": result.get("context_used", 0),
                        "multi_hop": result.get("metadata", {}).get("multi_hop_used", False)
                    }
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                    "citations": [],
                    "metadata": {}
                })
        
        # Clear input and rerun
        st.rerun()


if __name__ == "__main__":
    main()
