import streamlit as st
from rag_pipeline import load_and_index_pdfs, retrieve_context
from llm_handler import load_llm, build_prompt, get_answer

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# ─── Session State ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "session_active" not in st.session_state:
    st.session_state.session_active = True

if "active_theme" not in st.session_state:
    st.session_state.active_theme = "Midnight"


THEMES = {
    "Midnight": {
        "bg": "#0b1020",
        "panel": "#131a2a",
        "surface": "#1b2438",
        "border": "#2d3a57",
        "text": "#ecf2ff",
        "muted": "#a6b5d1",
        "accent": "#4cc9f0",
        "accent_2": "#7b2cbf",
    },
    "Graphite": {
        "bg": "#111315",
        "panel": "#1a1d21",
        "surface": "#252931",
        "border": "#3b4252",
        "text": "#edf2f4",
        "muted": "#b0b8c5",
        "accent": "#06d6a0",
        "accent_2": "#118ab2",
    },
    "Neon Dusk": {
        "bg": "#140f1f",
        "panel": "#1d1430",
        "surface": "#2b1d47",
        "border": "#5a3d91",
        "text": "#f6f0ff",
        "muted": "#cdbdf1",
        "accent": "#f72585",
        "accent_2": "#4cc9f0",
    },
}


def apply_theme(theme_name: str) -> None:
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top right, {theme["surface"]} 0%, {theme["bg"]} 48%, {theme["bg"]} 100%);
                color: {theme["text"]};
            }}
            [data-testid="stHeader"] {{
                background: transparent;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {theme["panel"]} 0%, {theme["bg"]} 100%);
                border-right: 1px solid {theme["border"]};
            }}
            .stMarkdown, p, label, .stCaption, .stAlert {{
                color: {theme["text"]} !important;
            }}
            .stButton>button {{
                width: 100%;
                border-radius: 10px;
                border: 1px solid {theme["border"]};
                background: linear-gradient(135deg, {theme["accent_2"]} 0%, {theme["accent"]} 100%);
                color: white;
                font-weight: 600;
            }}
            .stChatMessage {{
                background-color: {theme["panel"]};
                border: 1px solid {theme["border"]};
                border-radius: 12px;
            }}
            [data-testid="stTextInputRootElement"], [data-testid="stChatInput"] {{
                border-radius: 10px;
            }}
            .status-card {{
                background: {theme["panel"]};
                border: 1px solid {theme["border"]};
                border-radius: 12px;
                padding: 12px 14px;
                margin-bottom: 12px;
            }}
            .status-card .k {{
                color: {theme["muted"]};
                font-size: 0.9rem;
            }}
            .status-card .v {{
                color: {theme["text"]};
                font-weight: 700;
                font-size: 1.05rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ─── Sidebar: PDF Upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Control Center")
    selected_theme = st.selectbox("Dark theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.active_theme))
    st.session_state.active_theme = selected_theme

apply_theme(st.session_state.active_theme)

st.title("🧠 PDF RAG Studio")
st.caption("Upload PDFs, build the index, and ask precision questions. Type 'exit' to close the session.")

with st.sidebar:
    st.subheader("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🛠️ Build Vector Index", use_container_width=True):
        with st.spinner("Reading and indexing PDFs..."):
            try:
                st.session_state.vectorstore = load_and_index_pdfs(uploaded_files)
            except Exception as exc:
                st.session_state.vectorstore = None
                st.error(f"❌ Failed to process PDFs: {exc}")
            else:
                st.success(f"✅ Indexed {len(uploaded_files)} PDF file(s).")

    docs_count = len(uploaded_files) if uploaded_files else 0
    index_status = "Ready" if st.session_state.vectorstore else "Not Ready"
    model_status = "Ready" if st.session_state.llm else "Not Ready"

    st.markdown(
        f"""
        <div class="status-card">
            <div class="k">🗂️ Files queued</div>
            <div class="v">{docs_count}</div>
        </div>
        <div class="status-card">
            <div class="k">🔎 Index status</div>
            <div class="v">{index_status}</div>
        </div>
        <div class="status-card">
            <div class="k">🧠 Model status</div>
            <div class="v">{model_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.vectorstore:
        st.info("✅ Documents indexed. You can now ask questions.")

    st.divider()

    # Load LLM button
    if st.button("🚀 Load LLaMA Model", use_container_width=True):
        with st.spinner("Loading LLaMA model (this may take a minute)..."):
            try:
                st.session_state.llm = load_llm()
            except Exception as exc:
                st.session_state.llm = None
                st.error(f"❌ Failed to load model: {exc}")
            else:
                st.success("✅ LLaMA model is ready.")

    st.divider()

    # Reset chat
    if st.button("🧹 Reset Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_active = True
        st.rerun()

st.markdown("### ✨ Quick Starters")
quick_cols = st.columns(3)
quick_prompts = [
    "Summarize the key ideas from all uploaded PDFs.",
    "List important definitions and their sources.",
    "Extract action items and deadlines mentioned in the documents.",
]
for idx, prompt in enumerate(quick_prompts):
    if quick_cols[idx].button(f"Prompt {idx + 1}", use_container_width=True):
        st.session_state["quick_prompt"] = prompt

# ─── Chat Display ─────────────────────────────────────────────────────────────
if not st.session_state.session_active:
    st.warning("🛑 Session closed. Use 'Reset Conversation' in the sidebar to restart.")
else:
    # Render existing chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])

    # ─── Chat Input ──────────────────────────────────────────────────────────
    default_input = st.session_state.pop("quick_prompt", "")
    user_input = st.chat_input("Ask a question about your PDFs... (type 'exit' to end)")
    if default_input and not user_input:
        user_input = default_input

    if user_input:
        # Exit condition
        if user_input.strip().lower() == "exit":
            st.session_state.session_active = False
            with st.chat_message("assistant"):
                st.write("🛑 Session closed. Thanks for using PDF RAG Studio.")
            st.rerun()

        # Guard: PDFs must be uploaded
        if not st.session_state.vectorstore:
            st.warning("⚠️ Upload PDFs and build the vector index first.")
            st.stop()

        # Guard: LLM must be loaded
        if not st.session_state.llm:
            st.warning("⚠️ Load the LLaMA model from the sidebar first.")
            st.stop()

        # Show user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context = retrieve_context(st.session_state.vectorstore, user_input)
                    prompt = build_prompt(context, st.session_state.chat_history, user_input)
                    answer = get_answer(st.session_state.llm, prompt)
                except Exception as exc:
                    st.error(f"❌ Failed to generate answer: {exc}")
                    st.stop()
            st.write(answer)

        # Append to chat history (no repetition — one write per turn)
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": answer
        })
