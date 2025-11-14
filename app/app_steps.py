# app_steps.py — Step-by-step RAG configurator + Chatbot
import os, io, time, json, tempfile, csv
import logging, threading
from dotenv import load_dotenv
from typing import List, Dict
from dataclasses import dataclass
import streamlit as st
import pandas as pd
from utils.ui_helpers import openai_api_key_widget, get_openai_api_key

# Optional heavy libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    import faiss
except Exception:
    faiss = None

# Load environment variables from .env (if present)
load_dotenv()

st.set_page_config(page_title="DocuBot Studio — Setup Steps", layout="wide")
# Disable telemetry at runtime (avoids external webhook errors in console)
try:
    st.set_option('browser.gatherUsageStats', False)
except Exception:
    pass

# -------------------------
# Data classes + helpers
# -------------------------
@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    embedding: List[float] = None
    meta: dict = None

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []          # List[Chunk]
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "emb_model_name" not in st.session_state:
    st.session_state["emb_model_name"] = None
if "llm_choice" not in st.session_state:
    st.session_state["llm_choice"] = None
if "completed_steps" not in st.session_state:
    st.session_state["completed_steps"] = {f"step{i}": False for i in range(1,9)}
if "verif_log" not in st.session_state:
    st.session_state["verif_log"] = []
if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "do_chunk" not in st.session_state:
    st.session_state["do_chunk"] = False
if "chunk_in_progress" not in st.session_state:
    st.session_state["chunk_in_progress"] = False
if "chunk_params" not in st.session_state:
    st.session_state["chunk_params"] = {"size": 400, "overlap": 50}

# -------------------------
# Loading overlay (centered image + transparent background)
# -------------------------
def render_loading_overlay():
    st.markdown(
        """
        <style>
        .cs-overlay { position: fixed; inset: 0; background: rgba(255,255,255,0.6); z-index: 10000; display:flex; align-items:center; justify-content:center; }
        .cs-overlay .emoji { font-size: 84px; opacity: 0.95; animation: pedal 1.1s linear infinite; display:inline-block; }
        @keyframes pedal {
          0% { transform: translateY(0) translateX(0); }
          25% { transform: translateY(-4px) translateX(2px); }
          50% { transform: translateY(0) translateX(4px); }
          75% { transform: translateY(4px) translateX(2px); }
          100% { transform: translateY(0) translateX(0); }
        }
        html, body { overflow: hidden !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cs-overlay"><div class="emoji">🚴‍♂️</div></div>',
        unsafe_allow_html=True,
    )


# -------------------------
# Utility: chunking (define before background task uses it)
# -------------------------
def chunk_text(text: str, chunk_size: int, overlap: int):
    out, i, N = [], 0, len(text)
    while i < N:
        end = min(N, i + chunk_size)
        out.append(text[i:end].strip())
        i = end - overlap
        if i < 0:
            i = 0
        if i >= N:
            break
    return [c for c in out if len(c) > 20]


# -------------------------
# Debug + background threading helpers for chunking
# -------------------------
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    _h.setFormatter(_fmt)
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)

def _safe_run_chunking_task_threaded(chunk_size: int, overlap: int):
    try:
        _logger.info("Chunking(thread): starting")
        st.session_state["chunk_in_progress"] = True
        st.session_state["chunk_error"] = None
        st.session_state["chunk_progress"] = 0

        # Use existing raw texts already loaded in session
        raw_texts = [c.text for c in st.session_state.get("chunks", [])]
        if not raw_texts:
            raise ValueError("No documents available to chunk. Please complete Step 1 and upload PDFs.")
        total_len = sum(len(t or "") for t in raw_texts)
        if total_len > 2000000:
            raise ValueError("Input too large to chunk at once. Please upload fewer/smaller PDFs or reduce content size.")

        new_chunks = []
        for idx, t in enumerate(raw_texts):
            cs = chunk_text(t or "", chunk_size, overlap)
            for s in cs:
                cid = len(new_chunks)
                new_chunks.append(Chunk(chunk_id=cid, text=s, source="uploaded_docs"))
            # progress up to 95%, remaining for finalize
            st.session_state["chunk_progress"] = min(95, int((idx + 1) / max(1, len(raw_texts)) * 95))
            time.sleep(0.005)

        st.session_state["chunks"] = new_chunks
        st.session_state["chunk_progress"] = 100
        _logger.info(f"Chunking(thread): finished, created {len(new_chunks)} chunks")
    except Exception as e:
        st.session_state["chunk_error"] = repr(e)
        _logger.exception("Chunking(thread) failed")
    finally:
        st.session_state["chunk_in_progress"] = False

def start_chunking_in_background(chunk_size: int, overlap: int):
    if st.session_state.get("chunk_in_progress"):
        _logger.info("Chunking already in progress; ignoring start request")
        return None
    st.session_state["chunk_error"] = None
    t = threading.Thread(target=_safe_run_chunking_task_threaded, args=(chunk_size, overlap), daemon=True)
    t.start()
    _logger.info("Chunking: background thread started")
    return t

def render_chunking_debug_panel():
    st.markdown("### DEBUG: Chunking internals (temporary)")
    keys = [
        "do_chunk", "chunk_in_progress", "chunk_error", "chunk_progress",
        "chunk_params"
    ]
    debug = {k: st.session_state.get(k, "<missing>") for k in keys}
    debug["chunks_count"] = len(st.session_state.get("chunks", []))
    st.json(debug)


# -------------------------
# Background task triggers (two-phase pattern)
# -------------------------
def _run_chunking_task():
    # Uses params saved in session by Step 2 UI
    try:
        size = int(st.session_state["chunk_params"].get("size", 400))
        overlap = int(st.session_state["chunk_params"].get("overlap", 50))
        raw_texts = [c.text for c in st.session_state["chunks"]]
        # Validate inputs
        if not raw_texts:
            raise ValueError("No documents available to chunk. Please complete Step 1 and upload PDFs.")
        if not any((t or '').strip() for t in raw_texts):
            raise ValueError("Uploaded documents contain no extractable text.")
        total_len = sum(len(t or "") for t in raw_texts)
        if total_len > 2000000:
            raise ValueError("Input too large to chunk at once. Please upload fewer/smaller PDFs or reduce content size.")

        new_chunks = []
        base_id = 0
        for t in raw_texts:
            cs = chunk_text(t or "", size, overlap)
            for s in cs:
                cid = base_id + len(new_chunks)
                new_chunks.append(Chunk(chunk_id=cid, text=s, source="uploaded_docs"))

        st.session_state["chunks"] = new_chunks
        st.session_state["do_chunk"] = False
        st.session_state["chunk_in_progress"] = False
        st.session_state["busy"] = False
        st.toast(f"Created {len(new_chunks)} chunks.", icon="✅")
        st.rerun()
    except Exception as e:
        # Ensure flags are reset and notify user
        st.session_state["do_chunk"] = False
        st.session_state["chunk_in_progress"] = False
        st.session_state["busy"] = False
        st.error(f"Chunking failed: {e}")
        st.toast("Chunking failed. See error above.", icon="❌")
        st.rerun()


# Execute background chunking using a stable three-phase pattern
# 1) Button sets do_chunk=True then reruns
# 2) On next run, we only mark chunk_in_progress=True, render overlay, and rerun to let browser paint
# 3) On the following run, we perform the heavy work, then rerun to show results
if st.session_state.get("do_chunk") and not st.session_state.get("chunk_in_progress"):
    st.session_state["busy"] = True
    st.session_state["chunk_in_progress"] = True
    render_loading_overlay()
    st.rerun()
elif st.session_state.get("chunk_in_progress"):
    st.session_state["busy"] = True
    render_loading_overlay()
    _run_chunking_task()

# -------------------------
# Utility functions
# -------------------------
def parse_pdf_bytes(pdf_bytes: bytes):
    if not pdfplumber:
        st.error("pdfplumber not installed.")
        return []
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages

def embed_texts_sentence_transformer(texts: List[str], model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers for local embeddings.")
    model = st.session_state.get("local_emb_model")
    if model is None or getattr(model, "name", "") != model_name:
        model = SentenceTransformer(model_name)
        st.session_state["local_emb_model"] = model
    arr = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return arr.tolist()

def embed_texts_openai(texts: List[str], model="text-embedding-3-small"):
    if not OpenAI:
        raise RuntimeError("Install openai package.")
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("OpenAI API key missing — set it in Step 3 to use OpenAI embeddings.")
    client = OpenAI(api_key=key)
    try:
        embeddings = []
        for t in texts:
            resp = client.embeddings.create(input=t, model=model)
            embeddings.append(resp.data[0].embedding)
        return embeddings
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "429" in msg:
            st.error("OpenAI rate limit or quota exceeded. Switch to a local embedding model in Step 3 or update your OpenAI plan.")
            return []
        st.error(f"OpenAI embeddings failed: {e}")
        return []

def build_faiss_index(embeddings):
    if faiss is None or np is None:
        raise RuntimeError("Install faiss-cpu and numpy")
    xb = np.array(embeddings).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    return index

def retrieve_in_memory(query: str, top_k: int=5):
    # default fallback: naive ngram or chunk similarity - but prefer using FAISS if built
    if not st.session_state.get("faiss_index"):
        # improved fallback: keyword-overlap scoring across tokens
        import re
        stop = {
            "the","a","an","and","or","is","are","to","of","in","on","for","with","how","should","be","by","do","does","can","could","would","what","when","where","why","which"
        }
        toks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in stop]
        qset = set(toks)
        scored = []
        for c in st.session_state.get("chunks", [])[:500]:
            words = set(re.findall(r"[a-z0-9]+", (c.text or "").lower()))
            ov = len(qset & words)
            if ov > 0:
                scored.append((ov, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, c in scored[:top_k]:
            out.append({"chunk_id": c.chunk_id, "score": float(score), "text": c.text, "source": c.source})
        return out
    # with FAISS: embed query using the selected embedding model
    emb_name = st.session_state.get("emb_model_name")
    q_emb = None
    if emb_name and emb_name.startswith("openai"):
        q_emb = embed_texts_openai([query])[0]
    else:
        q_emb = embed_texts_sentence_transformer([query], emb_name)[0]
    vec = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(vec)
    D,I = st.session_state["faiss_index"].search(vec, top_k)
    results=[]
    for score, idx in zip(D[0], I[0]):
        if idx<0: continue
        c = st.session_state["chunks"][int(idx)]
        results.append({"chunk_id": c.chunk_id, "score": float(score), "text": c.text, "source": c.source})
    return results

def generate_answer_openai(question, retrieved, model_choice, temperature=0.0):
    if not OpenAI:
        raise RuntimeError("OpenAI client not installed.")
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("OpenAI API key missing — set it in Step 3 to use OpenAI LLMs.")
    client = OpenAI(api_key=key)
    context=""
    for r in retrieved:
        context += f"[chunk {r['chunk_id']} | score={r['score']}]\n{r['text']}\n\n---\n"
    system="You are a compliance assistant. Answer concisely in up to 3 bullets. After each bullet include source chunk ids in brackets."
    user=f"Context:\n{context}\n\nQuestion: {question}\n\nRequirements: 3 bullets max, cite chunk ids, if unsupported say 'No supporting policy found.'"
    try:
        resp = client.chat.completions.create(
            model=model_choice,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=512,
        )
        text=(resp.choices[0].message.content or "").strip()
        import re
        ids = [int(s) for s in re.findall(r"chunk\s*(\d+)", text, flags=re.IGNORECASE)]
        return {"answer": text, "sources": ids, "raw": {"id": resp.id}}
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "429" in msg:
            st.error("OpenAI rate limit or quota exceeded. Use a different model or retry later.")
        else:
            st.error(f"OpenAI chat failed: {e}")
        return {"answer": "[OpenAI error]", "sources": [], "raw": {"error": msg}}

# -------------------------
# Sidebar: steps checklist
# -------------------------
st.sidebar.title("Configuration Steps")
for i, label in enumerate([
    "Upload PDF(s)",
    "Chunk & Preview",
    "Embeddings & Model Selection",
    "Build Index & Retrieval",
    "Prompt Template & LLM Selection",
    "Run Fine-Tuning Experiments",
    "Run Verifiability Checks",
    "Launch Chatbot"
], start=1):
    k=f"step{i}"
    completed = st.session_state["completed_steps"].get(k, False)
    st.session_state["completed_steps"][k] = st.sidebar.checkbox(f"{i}. {label}", value=completed, key=f"chk_{k}")

# -------------------------
# Main UI: Step content
# -------------------------
st.title("DocuBot Studio — Stepwise Configuration")
tab = st.radio("Go to step", [f"Step {i}" for i in range(1,9)], index=0)

# STEP 1: Upload
if tab == "Step 1":
    st.header("Step 1 — Upload PDF(s)")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.session_state["chunks"] = []
        total_pages=0
        added=0
        for f in uploaded:
            data=f.read()
            pages = parse_pdf_bytes(data) if pdfplumber else [""]
            txt = "\n\n".join(pages)
            start_id = len(st.session_state["chunks"])
            chs = chunk_text(txt, 10000, 0)  # added big chunk for raw doc storage
            for i, t in enumerate(chs):
                cid=start_id+i
                st.session_state["chunks"].append(Chunk(chunk_id=cid, text=t, source=f.name))
                added+=1
            total_pages += len(pages)
        st.success(f"Indexed {len(uploaded)} documents, {added} raw-doc chunks, {total_pages} pages.")
        st.write("Indexed documents:")
        st.write(list(set([c.source for c in st.session_state["chunks"]])))

    if st.button("Mark Step 1 Complete"):
        st.session_state["completed_steps"]["step1"]=True
        st.success("Step 1 marked complete.")

# STEP 2: Chunk & Preview
if tab=="Step 2":
    st.header("Step 2 — Chunk & Preview")
    chunk_size = st.number_input("Chunk size (chars)", value=400, min_value=200, max_value=3000, step=50)
    overlap = st.number_input("Overlap (chars)", value=50, min_value=0, max_value=500, step=10)
    if st.button("Run chunking on current docs"):
        # Validate before triggering background work
        have_raw = bool(st.session_state.get("chunks")) and any(
            (c.text or '').strip() for c in st.session_state.get("chunks", [])
        )
        if not have_raw:
            st.warning("No documents with text to chunk. Please complete Step 1 first.")
        else:
            # Start non-blocking background chunking (preferred)
            st.session_state["chunk_params"] = {"size": int(chunk_size), "overlap": int(overlap)}
            start_chunking_in_background(int(chunk_size), int(overlap))
            st.info("Background chunking started. Use the debug panel below to monitor progress.")
    # Optional debug background run that won't block Streamlit's event loop
    if st.button("Run chunking (background debug)"):
        start_chunking_in_background(int(chunk_size), int(overlap))
        st.info("Background chunking started. Use the debug panel below to monitor progress.")
    # preview
    num_preview = st.slider("Preview N chunks", 1, 20, 5)
    if st.session_state["chunks"]:
        preview = st.session_state["chunks"][:num_preview]
        for c in preview:
            st.markdown(f"**Chunk {c.chunk_id}** (source: {c.source})")
            st.write(c.text[:600] + ("..." if len(c.text)>600 else ""))
    else:
        st.info("No chunks found. Run Step 1 first or click run chunking.")
    # Show debug panel at the end of Step 2
    render_chunking_debug_panel()
    # Status and error display for background worker
    if st.session_state.get("chunk_in_progress"):
        st.info("Chunking in progress — running in background. UI remains responsive.")
    elif st.session_state.get("chunk_error"):
        st.error("Chunking failed (see details below).")
        st.code(st.session_state.get("chunk_error"), language="text")
    elif st.session_state.get("chunks"):
        st.success(f"Chunking complete. {len(st.session_state['chunks'])} chunks available.")
    if st.button("Mark Step 2 Complete"):
        st.session_state["completed_steps"]["step2"]=True
        st.success("Step 2 marked complete.")

# STEP 3: Embeddings & Model Selection
if tab=="Step 3":
    st.header("Step 3 — Embeddings & Model Selection")
    # API key input widget (session-only by default; optional .env persistence)
    openai_api_key_widget(show_demo_toggle=True)
    emb_choice = st.selectbox(
        "Embedding source",
        [
            "all-MiniLM-L6-v2",
            "paraphrase-mpnet-base-v2",
            "intfloat/multilingual-e5-small",
            "openai-embedding-3-small",
        ],
    )
    st.session_state["emb_model_name"] = emb_choice
    if st.button("Compute embeddings for all chunks (may be slow)"):
        texts = [c.text for c in st.session_state["chunks"]]
        with st.spinner("Computing embeddings..."):
            if emb_choice.startswith("openai"):
                if OpenAI is None:
                    st.error("OpenAI client not installed. Install 'openai' package and retry.")
                    embs = []
                elif not get_openai_api_key():
                    st.error("OpenAI API key missing — set it in Step 3 to use OpenAI embeddings.")
                    embs = []
                else:
                    model_name = "text-embedding-3-small" if "embedding-3-small" in emb_choice else emb_choice
                    embs = embed_texts_openai(texts, model=model_name)
            else:
                embs = embed_texts_sentence_transformer(texts, emb_choice)
        # attach embeddings
        for c,e in zip(st.session_state["chunks"], embs):
            c.embedding = e
        st.success("Embeddings computed and attached to chunks.")
        st.write(f"Vector dim: {len(embs[0]) if embs else 'unknown'}")
    if st.button("Mark Step 3 Complete"):
        st.session_state["completed_steps"]["step3"]=True
        st.success("Step 3 marked complete.")

# STEP 4: Build index & Retrieval
if tab=="Step 4":
    st.header("Step 4 — Build Index & Retrieval Test")
    if st.button("Build FAISS index"):
        embs = [c.embedding for c in st.session_state["chunks"] if c.embedding is not None]
        if not embs:
            st.error("No embeddings found. Run Step 3.")
        else:
            st.session_state["faiss_index"] = build_faiss_index(embs)
            st.success("FAISS index built.")
    # ---------- Enhanced retrieval testers (single + batch) ----------
    DEFAULT_SAMPLE_QUESTIONS = [
        "What is Flykite Airlines' leave accrual policy?",
        "Who is responsible for HR grievances?",
        "How to apply for maternity leave?",
        "What are the employee code of conduct guidelines?",
        "What is the vacation carry-over limit?",
    ]

    def read_questions_from_uploaded(file_obj) -> list:
        try:
            content = io.TextIOWrapper(file_obj, encoding="utf-8").read()
        except Exception:
            content = file_obj.read().decode("utf-8")
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            return []
        if len(lines) == 1 and (",") in lines[0] and not lines[0].lower().startswith("question"):
            parts = [p.strip() for p in lines[0].split(",") if p.strip()]
            if parts:
                return parts
        return lines

    def parse_retrieval_results(results, top_k: int = 5):
        parsed = []
        for r in (results or []):
            item = {"chunk_text": None, "source": None, "score": None}
            if isinstance(r, dict):
                item["chunk_text"] = r.get("chunk_text") or r.get("text") or r.get("content") or r.get("chunk")
                item["source"] = r.get("source") or r.get("doc_id") or r.get("meta") or r.get("filename")
                item["score"] = r.get("score") or r.get("dist") or r.get("similarity")
            elif isinstance(r, (list, tuple)):
                if len(r) == 1:
                    item["chunk_text"] = r[0]
                elif len(r) == 2:
                    item["chunk_text"], item["score"] = r
                elif len(r) >= 3:
                    item["source"], item["chunk_text"], item["score"] = r[0], r[1], r[2]
            else:
                for attr in ("chunk_text", "text", "content", "chunk"):
                    if hasattr(r, attr):
                        item["chunk_text"] = getattr(r, attr)
                        break
                for attr in ("source", "doc_id", "meta", "filename"):
                    if hasattr(r, attr):
                        item["source"] = getattr(r, attr)
                        break
                for attr in ("score", "dist", "similarity"):
                    if hasattr(r, attr):
                        item["score"] = getattr(r, attr)
                        break
                if item["chunk_text"] is None:
                    try:
                        item["chunk_text"] = str(r)
                    except Exception:
                        item["chunk_text"] = "<unparseable result>"
            parsed.append(item)
        return parsed[:top_k]

    st.subheader("Test queries (single or batch)")
    if "sample_questions" not in st.session_state:
        st.session_state["sample_questions"] = DEFAULT_SAMPLE_QUESTIONS.copy()

    col1, col2 = st.columns([3,1])
    with col1:
        input_mode = st.radio(
            "Provide test queries:",
            ("Use defaults", "Upload questions file", "Paste/Enter manually"),
            index=0,
        )
    with col2:
        st.caption("Formats: .txt (one per line) or CSV (lines or comma-separated)")

    uploaded = None
    if input_mode == "Upload questions file":
        uploaded = st.file_uploader(
            "Upload a .txt or .csv file with one question per line",
            type=["txt", "csv"],
            accept_multiple_files=False,
        )
        if uploaded:
            try:
                parsed = read_questions_from_uploaded(uploaded)
                if parsed:
                    st.session_state["sample_questions"] = parsed
                    st.success(f"Loaded {len(parsed)} questions from file.")
                else:
                    st.warning("Uploaded file parsed to 0 questions.")
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")

    if input_mode == "Paste/Enter manually":
        txt = st.text_area("Paste one question per line", height=160)
        if st.button("Save pasted questions"):
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if lines:
                st.session_state["sample_questions"] = lines
                st.success(f"Saved {len(lines)} questions for testing.")
            else:
                st.warning("No valid questions found in pasted text.")

    st.markdown("**Current sample questions (editable)**")
    questions_container = st.empty()
    with questions_container.container():
        for i, q in enumerate(st.session_state["sample_questions"]):
            new_q = st.text_input(f"Q{i+1}", value=q, key=f"q_edit_{i}")
            st.session_state["sample_questions"][i] = new_q

    if st.button("Add a new blank sample question"):
        st.session_state["sample_questions"].append("")

    if st.button("Save sample questions (trim empties)"):
        st.session_state["sample_questions"] = [q for q in st.session_state["sample_questions"] if q and q.strip()]
        st.success("Saved.")

    st.divider()

    st.markdown("### Single-query tester")
    query_col1, query_col2 = st.columns([3,1])
    with query_col1:
        selected = st.selectbox(
            "Select sample question (or type a custom query below)",
            options=["-- choose sample --"] + st.session_state["sample_questions"],
            index=0,
            key="sample_select",
        )
        custom_query = st.text_input("Or type your own query here", value="", placeholder="Type a query and press Run")
    with query_col2:
        top_k_single = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
        run_single = st.button("Run retrieval test")

    if run_single:
        q = custom_query.strip() if custom_query.strip() else (selected if selected != "-- choose sample --" else "")
        if not q:
            st.warning("Enter a query or select a sample question.")
        else:
            with st.spinner("Running retrieval..."):
                t0 = time.time()
                try:
                    results = retrieve_in_memory(q, top_k=int(top_k_single))
                except Exception as e:
                    st.error(f"retrieve_in_memory raised an error: {e}")
                    results = []
                elapsed = time.time() - t0
            parsed = parse_retrieval_results(results, top_k=int(top_k_single))
            st.success(f"Retrieved {len(parsed)} results in {elapsed:.2f}s")
            for i, r in enumerate(parsed):
                st.markdown(f"**Result #{i+1}** — Score: {r.get('score')}")
                st.write(r.get("chunk_text"))
                if r.get("source"):
                    st.caption(f"source: {r.get('source')}")

    st.divider()

    st.markdown("### Batch test (run ALL sample questions)")
    batch_top_k = st.number_input("Batch Top-k", min_value=1, max_value=20, value=3, step=1, key="batch_top_k")
    run_batch = st.button("Run Batch Test")

    if run_batch:
        qlist = [q for q in st.session_state["sample_questions"] if q and q.strip()]
        if not qlist:
            st.warning("No sample questions available. Add/copy/paste some questions first.")
        else:
            summary = []
            full_results = {}
            with st.spinner(f"Running batch retrieval for {len(qlist)} questions..."):
                for q in qlist:
                    t0 = time.time()
                    try:
                        res = retrieve_in_memory(q, top_k=int(batch_top_k))
                    except Exception as e:
                        res = []
                        st.error(f"Error while retrieving for query: {q[:80]}... -> {e}")
                    t_elapsed = time.time() - t0
                    parsed = parse_retrieval_results(res, top_k=int(batch_top_k))
                    top_score = parsed[0].get("score") if parsed else None
                    summary.append({"query": q, "retrieved": len(parsed), "top_score": top_score, "time_s": round(t_elapsed, 3)})
                    full_results[q] = parsed
            st.markdown("**Batch summary**")
            st.table(summary)
            st.markdown("**Details**")
            for q in qlist:
                with st.expander(q, expanded=False):
                    parsed = full_results.get(q, [])
                    if not parsed:
                        st.write("No results.")
                    for i, r in enumerate(parsed):
                        st.markdown(f"**#{i+1}** — score: {r.get('score')}")
                        st.write(r.get("chunk_text"))
                        if r.get("source"):
                            st.caption(f"source: {r.get('source')}")
    if st.button("Mark Step 4 Complete"):
        st.session_state["completed_steps"]["step4"]=True
        st.success("Step 4 marked complete.")

# STEP 5: Prompt template & LLM
if tab=="Step 5":
    st.header("Step 5 — Prompt Template & LLM Selection")
    llm_choice = st.selectbox("LLM model (MVP)", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-16k"])
    st.session_state["llm_choice"] = llm_choice
    template = st.selectbox("Prompt template", ["short_answer_template", "policy_quote_template", "step_by_step_template"])
    if template=="short_answer_template":
        prompt = "You are a compliance assistant. Answer in 3 bullets and cite chunk ids. Question: {question}"
    elif template=="policy_quote_template":
        prompt = "Quote the relevant policy excerpt verbatim and then provide a short interpretation. Cite chunk ids. Question: {question}"
    else:
        prompt = "Provide step-by-step action items to comply with policy. Cite chunk ids. Question: {question}"
    prompt = st.text_area("Edit prompt template", value=prompt, height=160)
    test_q = st.text_input("Test question for generator", value="What is the leave policy during probation?")
    if st.button("Run generator test"):
        retrieved = retrieve_in_memory(test_q, top_k=5)
        try:
            if OpenAI is None:
                st.error("OpenAI client not installed. Install 'openai' package and set OPENAI_API_KEY.")
                gen = {"answer":"[OpenAI not configured]","sources":[]}
            elif not get_openai_api_key():
                st.error("OpenAI API key missing — set it in Step 3 to use OpenAI LLMs.")
                gen = {"answer":"[OpenAI not configured]","sources":[]}
            else:
                gen = generate_answer_openai(test_q, retrieved, st.session_state["llm_choice"], temperature=0.0)
            st.markdown("**Generated answer**")
            st.write(gen["answer"])
            st.markdown("**Citations**: " + (", ".join([f"[chunk {s}]" for s in gen.get("sources",[])])))
            # log
            st.session_state["verif_log"].append({"timestamp":time.time(), "question":test_q, "answer_preview":gen["answer"][:400], "sources":gen.get("sources",[]), "top_score": retrieved[0]["score"] if retrieved else 0.0, "grounded": bool(gen.get("sources"))})
        except Exception as e:
            st.error(f"Error generating: {e}")
    if st.button("Mark Step 5 Complete"):
        st.session_state["completed_steps"]["step5"]=True
        st.success("Step 5 marked complete.")

# STEP 6: Experiments
if tab=="Step 6":
    st.header("Step 6 — Fine-tuning / Hyperparameter Experiments")
    st.write("Run the in-notebook experiment runner or use the app mock runner.")
    if st.button("Run mock experiments"):
        trials = [
            {"embedding":"all-MiniLM-L6-v2","chunk_size":250,"top_k":3,"avg_top_score":0.70,"grounded_pct":0.67},
            {"embedding":"all-MiniLM-L12-v2","chunk_size":400,"top_k":5,"avg_top_score":0.74,"grounded_pct":1.0},
            {"embedding":"paraphrase-mpnet-base-v2","chunk_size":400,"top_k":5,"avg_top_score":0.77,"grounded_pct":1.0},
        ]
        st.table(pd.DataFrame(trials))
        st.success("Mock experiments finished.")
    if st.button("Mark Step 6 Complete"):
        st.session_state["completed_steps"]["step6"]=True
        st.success("Step 6 marked complete.")

# STEP 7: Verifiability Checks
if tab=="Step 7":
    st.header("Step 7 — Verifiability Checks")
    if st.session_state["verif_log"]:
        df = pd.DataFrame(st.session_state["verif_log"])
        st.dataframe(df)
        if st.button("Download verifiability_log.csv"):
            st.download_button("Download", df.to_csv(index=False).encode("utf-8"), "verifiability_log.csv")
    else:
        st.info("No verifiability entries yet (run generator tests).")
    if st.button("Mark Step 7 Complete"):
        st.session_state["completed_steps"]["step7"]=True
        st.success("Step 7 marked complete.")

# STEP 8: Launch Chatbot
if tab=="Step 8":
    # ---------------- Step 8: Launch Chatbot ----------------
    # helper: simple fallback answer when LLM not configured
    def fallback_generate_answer(question, retrieved_chunks, top_k=3):
        """
        Simple fallback: if we have retrieved chunks, return the top one + pointer.
        """
        if not retrieved_chunks:
            return "I couldn't find relevant document excerpts to answer that. Try rewording your question."
        first = retrieved_chunks[0]
        if isinstance(first, dict):
            text = first.get("chunk_text") or first.get("text") or first.get("content") or str(first)
            source = first.get("source") or first.get("doc_id") or ""
        elif isinstance(first, (list, tuple)):
            text = first[1] if len(first) > 1 else str(first[0])
            source = first[0] if len(first) > 2 else ""
        else:
            text = str(first)
            source = ""
        answer = f"Here is an excerpt that may help:\n\n{text}\n\n(If this doesn't answer your question, try rephrasing or run the retrieval test.)"
        if source:
            answer += f"\n\nSource: {source}"
        return answer

    st.title("Step 8 — Launch Chatbot")

    # Decide mode: config vs chatbot (default config)
    mode = st.session_state.get("mode", "config")

    if mode == "config":
        # show configuration status JSON
        st.subheader("Configuration status:")
        st.json(st.session_state["completed_steps"])

        # UI for app name, tagline, welcome message
        st.markdown("---")
        col_name, col_tag = st.columns([2,3])
        with col_name:
            app_name = st.text_input("App name", value=st.session_state.get("launched_app_name","DocuBot Studio"))
        with col_tag:
            tagline = st.text_input("Tagline", value=st.session_state.get("launched_tagline","Transform documentation into AI assistants."))

        welcome_note = st.text_area("Welcome note (shown when chatbot opens)", value=st.session_state.get("launched_welcome","Hi — I'm your document assistant. Ask me anything about the uploaded docs!"), height=80)

        launch_col1, launch_col2 = st.columns([1,1])
        with launch_col1:
            if st.button("Launch Chatbot", key="launch_chatbot_btn"):
                st.session_state["mode"] = "chatbot"
                st.session_state["launched_app_name"] = app_name
                st.session_state["launched_tagline"] = tagline
                st.session_state["launched_welcome"] = welcome_note
                st.session_state.setdefault("chat_history", [])
        with launch_col2:
            if st.button("Reset Chatbot History", key="reset_chat_history_btn"):
                st.session_state.pop("chat_history", None)
                st.success("Chat history cleared.")

        st.markdown("----")
        st.caption("After launching, the configuration panel will be replaced with the interactive chatbot. Use the Close button in the chat header to return here.")

    elif mode == "chatbot":
        # Chat header
        st.markdown("---")
        st.markdown(f"## {st.session_state.get('launched_app_name','DocuBot Studio')}")
        st.markdown(f"**{st.session_state.get('launched_tagline','Transform documentation into AI assistants.')}**")
        close_col, spacer = st.columns([1,9])
        with close_col:
            if st.button("Close Chatbot", key="close_chatbot"):
                st.session_state["mode"] = "config"

        st.markdown("---")
        st.session_state.setdefault("chat_history", [])
        if not st.session_state["chat_history"]:
            st.session_state["chat_history"].append({"role":"bot","text": st.session_state.get("launched_welcome","Hi — I'm your document assistant. Ask me anything about the uploaded docs!")})

        for msg in st.session_state["chat_history"]:
            role = msg.get("role","bot")
            text = msg.get("text","")
            if hasattr(st, "chat_message"):
                with st.chat_message(role):
                    st.markdown(text)
            else:
                if role == "user":
                    st.markdown(f"> **You:** {text}")
                else:
                    st.markdown(f"**Assistant:** {text}")

        user_input = st.text_input("Type a message", key="chat_input", placeholder="Ask something about your uploaded documents...")
        send_col, dummy = st.columns([1,9])
        with send_col:
            if st.button("Send", key="send_msg_btn"):
                query = user_input.strip()
                if not query:
                    st.warning("Please type a question.")
                else:
                    st.session_state["chat_history"].append({"role":"user","text": query})
                    with st.spinner("Generating response..."):
                        try:
                            retrieved = retrieve_in_memory(query, top_k=5)
                        except Exception as e:
                            retrieved = []
                            st.error(f"Retrieval failed: {e}")

                        try:
                            answer_text = None
                            # Prefer centralized helper (session/env) to avoid secrets.toml errors
                            try:
                                key_present = bool(get_openai_api_key())
                            except Exception:
                                key_present = False
                            if key_present:
                                try:
                                    answer_obj = generate_answer_openai(query, retrieved, model_choice=st.session_state.get("llm_choice","gpt-4o-mini"), temperature=0.0)
                                    answer_text = answer_obj.get("answer") if isinstance(answer_obj, dict) else str(answer_obj)
                                except Exception as e:
                                    st.warning(f"LLM call failed, using fallback: {e}")
                                    answer_text = fallback_generate_answer(query, retrieved)
                            else:
                                answer_text = fallback_generate_answer(query, retrieved)
                        except NameError:
                            answer_text = fallback_generate_answer(query, retrieved)
                        except Exception as e:
                            st.error(f"Error while generating answer: {e}")
                            answer_text = fallback_generate_answer(query, retrieved)

                        st.session_state["chat_history"].append({"role":"bot","text": answer_text})

st.sidebar.markdown("---")
st.sidebar.info("Use the tabs to perform each step. Mark steps complete to indicate configuration readiness.")
