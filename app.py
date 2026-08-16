import hashlib
import html
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pipeline import InvestmentResearchRAG

st.set_page_config(
    page_title="Meridian | Investment Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {--ink:#10231f;--muted:#63726e;--line:#dce5e1;--paper:#f7f9f8;--accent:#0c7c66;--soft:#e8f4f0}
    .stApp{background:radial-gradient(circle at 82% 3%,rgba(12,124,102,.07),transparent 25rem),var(--paper);color:var(--ink)}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stMainBlockContainer"]{max-width:1120px;padding-top:2.5rem;padding-bottom:4rem}
    [data-testid="stSidebar"]{background:#10231f;border-right:1px solid rgba(255,255,255,.08)}
    [data-testid="stSidebar"] *{color:#edf4f1}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.055);border:1px dashed rgba(255,255,255,.28);border-radius:14px}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small{color:#b9c8c3!important}
    [data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.11)}
    h1,h2,h3{color:var(--ink);letter-spacing:-.025em} p{line-height:1.65}
    .brand-kicker{color:var(--accent);font-size:.73rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.55rem}
    .hero-title{color:var(--ink);font-family:Georgia,'Times New Roman',serif;font-size:clamp(2.25rem,5vw,4.25rem);font-weight:500;letter-spacing:-.045em;line-height:1.02;margin:0;max-width:850px}
    .hero-copy{color:var(--muted);font-size:1.03rem;max-width:690px;margin:1rem 0 1.8rem}
    .eyebrow{color:var(--muted);font-size:.72rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase}
    .document-card,.empty-card{background:rgba(255,255,255,.86);border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 35px rgba(16,35,31,.055);padding:1.25rem 1.35rem}
    .document-title{color:var(--ink);font-size:1.02rem;font-weight:760;margin:.25rem 0 .15rem}
    .document-meta{color:var(--muted);font-size:.87rem}
    .ready-dot{background:#17a77f;border-radius:50%;box-shadow:0 0 0 4px rgba(23,167,127,.13);display:inline-block;height:8px;margin-right:.45rem;width:8px}
    .metric-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem;margin-top:.8rem}
    .metric-cell{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.09);border-radius:11px;padding:.6rem}
    .metric-value{color:#fff;font-size:1rem;font-weight:750}.metric-label{color:#9fb3ad;font-size:.62rem;margin-top:.12rem}
    .citation{background:var(--soft);border:1px solid #cce5dc;border-radius:999px;color:#086552;display:inline-block;font-size:.76rem;font-weight:700;margin:.2rem .25rem .1rem 0;padding:.25rem .62rem}
    .footer-note{color:#81908b;font-size:.73rem;margin-top:2.5rem;text-align:center}
    [data-testid="stChatMessage"]{background:rgba(255,255,255,.84);border:1px solid var(--line);border-radius:17px;box-shadow:0 5px 22px rgba(16,35,31,.035);margin-bottom:.75rem;padding:.2rem .35rem}
    [data-testid="stChatMessage"] p{color:#213a34}
    [data-testid="stChatInput"]{background:#fff;border-color:#cfdcd7;box-shadow:0 9px 28px rgba(16,35,31,.09)}
    .stButton>button{border-radius:10px;font-weight:680;transition:all .16s ease}.stButton>button:hover{transform:translateY(-1px)}
    [data-testid="stSidebar"] .stButton>button[kind="primary"]{background:#20a482;border-color:#20a482;color:#fff}
    @media(max-width:700px){[data-testid="stMainBlockContainer"]{padding-top:1.4rem}.metric-strip{grid-template-columns:1fr}.hero-title{font-size:2.4rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {
    "messages": [], "rag": None, "document_name": None,
    "document_size_mb": None, "document_pages": None,
    "document_chunks": None, "document_hash": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_document():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def citation_pages(sources):
    pages = []
    for source in sources or []:
        page = source.get("page") if isinstance(source, dict) else source
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def render_citations(sources):
    pages = citation_pages(sources)
    if pages:
        chips = "".join(
            f'<span class="citation">Page {html.escape(str(page))}</span>'
            for page in pages
        )
        st.markdown(chips, unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        """<div style="padding:.4rem 0 .8rem"><div style="font-family:Georgia,serif;font-size:1.55rem;font-weight:600">◈ Meridian</div><div style="color:#9fb3ad;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase">Document intelligence</div></div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Research document")
    st.caption("Upload a text-based financial filing in PDF format.")
    uploaded_file = st.file_uploader("Financial PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_size_mb = len(upload_bytes) / (1024 * 1024)
        st.caption(f"{uploaded_file.name} · {upload_size_mb:.2f} MB")

        if st.button("Process document", type="primary", use_container_width=True):
            try:
                if not upload_bytes:
                    raise ValueError("The uploaded file is empty. Please select it again.")
                if not upload_bytes.lstrip().startswith(b"%PDF-"):
                    raise ValueError("The selected file is not a valid PDF.")

                safe_filename = Path(uploaded_file.name).name
                content_hash = hashlib.sha256(upload_bytes).hexdigest()
                upload_folder = PROJECT_ROOT / "data" / "uploads"
                upload_folder.mkdir(parents=True, exist_ok=True)
                saved_path = upload_folder / f"{content_hash[:16]}_{safe_filename}"
                temporary_path = saved_path.with_suffix(saved_path.suffix + ".tmp")

                with st.status("Preparing document…", expanded=True) as status:
                    st.write("Validating and securing upload")
                    with open(temporary_path, "wb") as file:
                        file.write(upload_bytes)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary_path, saved_path)
                    if saved_path.stat().st_size != len(upload_bytes):
                        raise IOError("The PDF was not saved completely.")

                    st.write("Extracting text and creating semantic chunks")
                    st.write("Building search index and loading reranker")
                    rag = InvestmentResearchRAG(saved_path)
                    st.session_state.rag = rag
                    st.session_state.document_name = safe_filename
                    st.session_state.document_size_mb = upload_size_mb
                    st.session_state.document_pages = len(rag.pages)
                    st.session_state.document_chunks = len(rag.chunks)
                    st.session_state.document_hash = content_hash
                    st.session_state.messages = []
                    status.update(label="Document ready", state="complete", expanded=False)
                st.success("Analysis workspace is ready.")
            except Exception as error:
                st.session_state.rag = None
                st.error(f"Processing failed: {error}")
                with st.expander("Technical details"):
                    st.exception(error)

    st.divider()
    if st.session_state.rag is not None:
        st.markdown("<span class='ready-dot'></span> **Ready for research**", unsafe_allow_html=True)
        st.caption(st.session_state.document_name)
        left, right = st.columns(2)
        left.metric("Pages", st.session_state.document_pages)
        right.metric("Chunks", st.session_state.document_chunks)
        st.caption(f"{st.session_state.document_size_mb:.2f} MB · ID {st.session_state.document_hash[:8]}")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Remove document", use_container_width=True):
            reset_document()
            st.rerun()
    else:
        st.caption("○ No document currently indexed")

    st.divider()
    with st.expander("Methodology & evaluation"):
        st.caption("Semantic chunking → dense retrieval (15) → cross-encoder reranking (5) → grounded generation")
        st.markdown(
            """<div class="metric-strip"><div class="metric-cell"><div class="metric-value">0.713</div><div class="metric-label">FIXED MRR</div></div><div class="metric-cell"><div class="metric-value">0.854</div><div class="metric-label">SEMANTIC MRR</div></div><div class="metric-cell"><div class="metric-value">0.875</div><div class="metric-label">RERANKED MRR</div></div></div>""",
            unsafe_allow_html=True,
        )
        st.caption("Hit@5: 1.000 across evaluated configurations")


st.markdown('<div class="brand-kicker">Evidence-led investment research</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Ask better questions of financial filings.</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Turn dense reports into concise, source-grounded analysis. Every answer is retrieved from your document and linked back to its pages.</div>',
    unsafe_allow_html=True,
)

if st.session_state.rag is None:
    st.markdown(
        """<div class="empty-card"><div class="eyebrow">Start a research session</div><div class="document-title">Upload a financial PDF from the sidebar</div><div class="document-meta">The document is processed into searchable semantic evidence before questions are enabled.</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("### Built for diligence workflows")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Risk review**  \nIdentify operational, regulatory, and market risks.")
    col2.markdown("**Financial analysis**  \nTrace revenue, margins, and operating performance.")
    col3.markdown("**Evidence trail**  \nValidate conclusions against cited source pages.")
else:
    st.markdown(
        f"""<div class="document-card"><div class="eyebrow"><span class="ready-dot"></span>Active research document</div><div class="document-title">{html.escape(st.session_state.document_name)}</div><div class="document-meta">{st.session_state.document_pages} pages · {st.session_state.document_chunks} semantic chunks · Cross-encoder reranking active</div></div>""",
        unsafe_allow_html=True,
    )

    selected_question = None
    if not st.session_state.messages:
        st.markdown("#### Suggested research questions")
        suggestions = [
            "What are the most material risks discussed?",
            "How did revenue and margins change during the year?",
            "What does management say about competition?",
        ]
        for column, suggestion in zip(st.columns(3), suggestions):
            if column.button(suggestion, use_container_width=True):
                selected_question = suggestion

    st.markdown("#### Research conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_citations(message.get("sources"))

    typed_question = st.chat_input("Ask about performance, risks, strategy, or disclosures…")
    question = typed_question or selected_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.status("Analyzing document evidence…", expanded=False) as status:
                    result = st.session_state.rag.ask(question)
                    status.update(label="Analysis complete", state="complete")

                answer = result["answer"]
                source_pages = citation_pages(result.get("sources", []))
                st.markdown(answer)
                render_citations(source_pages)

                if result.get("sources"):
                    with st.expander("Review retrieved evidence"):
                        for source in result["sources"]:
                            st.markdown(f"**Page {source['page']}**")
                            preview = source.get("text", "").strip()
                            if preview:
                                st.caption(preview[:650] + ("…" if len(preview) > 650 else ""))
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "sources": source_pages,
                })
            except Exception as error:
                error_message = "I couldn’t complete that analysis. Please try again, or reprocess the document if the issue continues."
                st.error(error_message)
                with st.expander("Technical details"):
                    st.exception(error)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_message, "sources": [],
                })

st.markdown(
    '<div class="footer-note">AI-generated research support · Verify material conclusions against the original filing</div>',
    unsafe_allow_html=True,
)
