# -*- coding: utf-8 -*-
"""
GW University RAG Chatbot — Streamlit App
Deployment-ready: API key loaded from Streamlit Secrets automatically.
"""

import os
import re
import pickle
import tempfile
import warnings
import logging

# ── Suppress noisy torchvision / transformers warnings ───────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GW University Chatbot",
    page_icon="🎓",
    layout="wide",
)

# ─── Index file path (same folder as this script) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE  = os.path.join(SCRIPT_DIR, "gw_index.pkl")

# ─── Load API key: Streamlit Secrets first, then manual input ────────────────
# On Streamlit Cloud: key comes from Secrets (professor never sees this step)
# Running locally:    key is entered manually in the sidebar
def get_api_key_from_secrets():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        return key if key else ""
    except Exception:
        return ""

SECRETS_API_KEY = get_api_key_from_secrets()

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
.gw-header {
    background: linear-gradient(135deg, #002147 0%, #004080 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 16px;
    margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,33,71,0.25);
}
.gw-header h1 { font-family: 'Playfair Display', serif; font-size: 2.2rem; margin: 0 0 0.3rem 0; }
.gw-header p  { opacity: 0.8; font-size: 1rem; margin: 0; font-weight: 300; }
.chat-user {
    background: #002147; color: white; padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px; margin: 0.8rem 0 0.8rem auto;
    max-width: 72%; font-size: 0.95rem; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: block;
}
.chat-bot {
    background: #f8f9fa; color: #1a1a2e; padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px; margin: 0.8rem 0; max-width: 80%;
    font-size: 0.95rem; line-height: 1.65; border-left: 4px solid #002147;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.chat-bot p  { margin: 0.3rem 0; }
.chat-bot ul { margin: 0.4rem 0 0.4rem 1.2rem; padding: 0; }
.chat-bot li { margin-bottom: 0.25rem; }
.chat-bot strong { color: #002147; }
.source-tag {
    display: inline-block; background: #e8edf5; color: #002147;
    font-size: 0.72rem; padding: 2px 9px; border-radius: 12px;
    margin: 3px 3px 0 0; font-weight: 600;
}
.sources-row { margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid #dde4ef; }
.sidebar-section {
    background: #f0f4f9; border-radius: 10px;
    padding: 1rem; margin-bottom: 1rem; border: 1px solid #dde4ef;
}
.sidebar-section h4 {
    color: #002147; margin: 0 0 0.6rem 0; font-size: 0.85rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
}
.doc-stat    { font-size: 0.82rem; color: #444; margin: 2px 0; }
.badge-green { color: #1e7e34; font-weight: 600; }
.badge-red   { color: #c0392b; font-weight: 600; }
.badge-blue  { color: #0056b3; font-weight: 600; }
.welcome-card {
    background: linear-gradient(135deg, #f0f4fb, #e8edf8);
    border: 1px solid #ccd6e8; border-radius: 14px;
    padding: 2rem; text-align: center; color: #002147; margin-top: 2rem;
}
.welcome-card h3 { margin: 0 0 0.5rem 0; font-size: 1.3rem; }
.welcome-card p  { margin: 0; color: #555; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gw-header">
    <h1>🎓 GW University Chatbot</h1>
    <p>Ask me anything about GW programs, regulations, tuition, admissions, and academic policies.</p>
</div>
""", unsafe_allow_html=True)

# ─── Save / Load Index ────────────────────────────────────────────────────────
def save_index(chunks, sources, names, embeddings):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources,
                     "names": names, "embeddings": embeddings}, f)

def load_index():
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
            return data["chunks"], data["sources"], data.get("names", []), data["embeddings"]
        except Exception:
            pass
    return None, None, None, None

# ─── Session State ────────────────────────────────────────────────────────────
DEFAULTS = {
    "chat_history": [], "all_chunks": [], "chunk_sources": [],
    "embeddings": None, "embed_model": None, "groq_client": None,
    "pdfs_loaded": False, "pdf_names": [], "prefill_question": "",
    "index_checked": False, "api_key_active": "",
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Auto-load saved index ONCE per session ───────────────────────────────────
if not st.session_state.index_checked:
    st.session_state.index_checked = True
    if not st.session_state.pdfs_loaded:
        chunks, sources, names, embeddings = load_index()
        if chunks is not None:
            st.session_state.all_chunks    = chunks
            st.session_state.chunk_sources = sources
            st.session_state.pdf_names     = names
            st.session_state.embeddings    = embeddings
            st.session_state.pdfs_loaded   = True

# ─── Auto-init Groq client from secrets if available ─────────────────────────
if SECRETS_API_KEY and not st.session_state.groq_client:
    try:
        st.session_state.groq_client   = Groq(api_key=SECRETS_API_KEY)
        st.session_state.api_key_active = SECRETS_API_KEY
    except Exception:
        pass

# ─── PDF / Embedding helpers ──────────────────────────────────────────────────
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

def chunk_text_with_overlap(text, chunk_size=250, overlap=50):
    words = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index_from_uploads(uploaded_files):
    all_chunks, chunk_sources, pdf_names = [], [], []
    progress = st.progress(0, text="Extracting text from PDFs…")
    for i, f in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        chunks = chunk_text_with_overlap(extract_text_from_pdf(tmp_path))
        source_name = f.name.replace(".pdf","").replace("-"," ").replace("_"," ").title()
        all_chunks.extend(chunks)
        chunk_sources.extend([source_name] * len(chunks))
        pdf_names.append(f.name)
        os.unlink(tmp_path)
        progress.progress((i + 1) / len(uploaded_files), text=f"Processed: {f.name}")
    progress.empty()
    return all_chunks, chunk_sources, pdf_names

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_embed_model():
    if st.session_state.embed_model is None:
        st.session_state.embed_model = load_embed_model()
    return st.session_state.embed_model

def build_embeddings(chunks, model):
    return model.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

# ─── Semantic Search ──────────────────────────────────────────────────────────
def semantic_search(query, k=5, min_similarity=0.25):
    if st.session_state.embeddings is None or not st.session_state.all_chunks:
        return []
    emb    = st.session_state.embeddings
    model  = get_embed_model()
    q_vec  = model.encode([query], convert_to_numpy=True)
    norms  = np.linalg.norm(emb, axis=1, keepdims=True)
    q_norm = np.linalg.norm(q_vec)
    sims   = (emb @ q_vec.T).squeeze() / (norms.squeeze() * q_norm + 1e-10)
    top_idx = np.argsort(sims)[-k:][::-1]
    return [
        (st.session_state.all_chunks[idx], st.session_state.chunk_sources[idx], float(sims[idx]))
        for idx in top_idx if float(sims[idx]) >= min_similarity
    ]

def render_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    html_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("- ") or s.startswith("• "):
            html_lines.append(f"<li>{s[2:]}</li>")
        elif s:
            html_lines.append(f"<p>{s}</p>")
    html = "\n".join(html_lines)
    return re.sub(r'((<li>.*?</li>\n?)+)', r'<ul>\1</ul>', html)

# ─── Answer Generation ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful academic advisor chatbot for The George Washington University (GWU). "
    "Answer questions ONLY based on the context provided from GWU's official bulletin documents. "
    "If the context doesn't contain the answer, say: "
    "'I don't have enough information. Please contact GW directly.' "
    "Be concise, use bullet points for lists, and cite exact numbers for GPA, credits, and deadlines."
)

def generate_answer(question, context_results, chat_history):
    if not context_results:
        return "I couldn't find relevant information in the GW documents. Please contact GW directly at **gwu.edu**."
    context_block = "".join(
        f"[Context {i} — Source: {src}]\n{chunk}\n\n"
        for i, (chunk, src, _) in enumerate(context_results, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prev_q, prev_a in chat_history[-4:]:
        messages += [{"role": "user", "content": prev_q},
                     {"role": "assistant", "content": prev_a}]
    messages.append({"role": "user", "content":
        f"Use the following context from GW University documents to answer the question.\n\n"
        f"CONTEXT:\n{context_block}QUESTION: {question}"
    })
    response = st.session_state.groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, max_tokens=700)
    return response.choices[0].message.content

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    # ── API Key section ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><h4>🔑 Groq API Key</h4>', unsafe_allow_html=True)

    if SECRETS_API_KEY and st.session_state.groq_client:
        # Key loaded from Streamlit Secrets — hide the input box entirely
        st.markdown('<span class="badge-green">✅ API key loaded automatically</span>',
                    unsafe_allow_html=True)
        api_key = SECRETS_API_KEY
    else:
        # Running locally — show manual input
        api_key = st.text_input("Groq API Key", type="password",
                                 placeholder="gsk_…", label_visibility="collapsed")
        if api_key:
            try:
                st.session_state.groq_client   = Groq(api_key=api_key)
                st.session_state.api_key_active = api_key
                st.markdown('<span class="badge-green">✅ API key set</span>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<span class="badge-red">❌ {e}</span>', unsafe_allow_html=True)
        else:
            api_key = st.session_state.api_key_active  # keep previous value

    st.markdown("</div>", unsafe_allow_html=True)

    # ── PDF section ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><h4>📄 GW PDF Documents</h4>', unsafe_allow_html=True)

    index_exists = os.path.exists(INDEX_FILE)
    if st.session_state.pdfs_loaded and index_exists:
        st.markdown(
            '<span class="badge-green">✅ Knowledge base loaded — ready to answer!</span>',
            unsafe_allow_html=True)
        for name in st.session_state.pdf_names:
            st.markdown(f'<div class="doc-stat">📄 {name}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="doc-stat">🔢 {len(st.session_state.all_chunks):,} chunks indexed</div>',
            unsafe_allow_html=True)
    else:
        st.info("No saved index found. Upload GW PDFs below to build one.")

    uploaded_pdfs = st.file_uploader(
        "Upload GW bulletin PDFs", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_pdfs:
        label = "🔄 Re-process PDFs" if st.session_state.pdfs_loaded else "🚀 Process PDFs & Build Index"
        if st.button(label, use_container_width=True):
            with st.spinner("Building search index… (may take 1–2 mins)"):
                model  = load_embed_model()
                st.session_state.embed_model = model
                chunks, sources, names = build_index_from_uploads(uploaded_pdfs)
                embeddings = build_embeddings(chunks, model)
                st.session_state.all_chunks    = chunks
                st.session_state.chunk_sources = sources
                st.session_state.pdf_names     = names
                st.session_state.embeddings    = embeddings
                st.session_state.pdfs_loaded   = True
                save_index(chunks, sources, names, embeddings)
            st.success(f"✅ {len(chunks):,} chunks saved! Index persists across restarts.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Options ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><h4>🎛️ Options</h4>', unsafe_allow_html=True)
    top_k   = st.slider("Top-K chunks retrieved", 2, 10, 5)
    min_sim = st.slider("Min similarity threshold", 0.10, 0.60, 0.25, 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("♻️ Reset All", use_container_width=True):
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    # ── Sample Questions ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><h4>💡 Sample Questions</h4>', unsafe_allow_html=True)
    for q in [
        "Minimum GPA for graduate students?",
        "Full-time undergraduate tuition?",
        "What happens with an Incomplete grade?",
        "How to apply for a leave of absence?",
        "What engineering master programs exist?",
        "How to transfer credits to GW?",
        "Academic integrity policy?",
    ]:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.prefill_question = q
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Chat Display ─────────────────────────────────────────────────────────────
with st.container():
    if not st.session_state.chat_history:
        steps = []
        if not st.session_state.groq_client:
            steps.append("🔑 Enter your Groq API key in the sidebar")
        if not st.session_state.pdfs_loaded:
            steps.append("📄 Upload & process GW PDFs in the sidebar")
        if steps:
            st.markdown(
                '<div class="welcome-card"><h3>👋 Welcome to the GW Chatbot!</h3>'
                '<p>Complete these steps to get started:</p><br>'
                + "".join(f"<p>{s}</p>" for s in steps) + "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="welcome-card"><h3>✅ Ready to chat!</h3>'
                '<p>Type a question below or choose one from the sidebar.</p></div>',
                unsafe_allow_html=True)
    else:
        for user_q, bot_a, sources in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">{user_q}</div>', unsafe_allow_html=True)
            src_tags = "".join(
                f'<span class="source-tag">📄 {s} ({sc:.0%})</span>' for s, sc in sources)
            st.markdown(
                f'<div class="chat-bot">{render_markdown(bot_a)}'
                + (f'<div class="sources-row">{src_tags}</div>' if src_tags else "")
                + "</div>", unsafe_allow_html=True)

# ─── Input Bar ────────────────────────────────────────────────────────────────
prefill = st.session_state.prefill_question
if prefill:
    st.session_state.prefill_question = ""

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask a question", value=prefill,
            placeholder="e.g. What is the minimum GPA for graduate students?",
            label_visibility="collapsed")
    with cols[1]:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    if not st.session_state.groq_client:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    elif not st.session_state.pdfs_loaded:
        st.error("⚠️ Please upload and process GW PDFs first.")
    else:
        with st.spinner("Thinking…"):
            ctx    = semantic_search(user_input, k=top_k, min_similarity=min_sim)
            answer = generate_answer(
                user_input, ctx,
                [(q, a) for q, a, _ in st.session_state.chat_history])
            deduped = list(dict.fromkeys((src, round(sc, 2)) for _, src, sc in ctx))[:3]
            st.session_state.chat_history.append((user_input, answer, deduped))
        st.rerun()
