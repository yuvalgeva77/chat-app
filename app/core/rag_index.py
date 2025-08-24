# app/core/rag_index.py
from __future__ import annotations
import os, re, glob, json, pickle, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from app.core.config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DEFAULT
from app.core.logging_config import get_logger

logger = get_logger("rag-index")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    logger.error("sentence-transformers not installed. Install it for RAG.")

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Read fail {path}: {e}")
        return ""

def _read_pdf(path: str) -> str:
    try:
        import PyPDF2
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            return "\n".join([p.extract_text() or "" for p in r.pages])
    except Exception as e:
        logger.warning(f"PDF extraction failed for {path}: {e}")
        return ""

def _chunk(text: str, size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    out, i, step = [], 0, max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i+size])
        i += step
    return out

@dataclass
class DocChunk:
    doc_id: str
    title: str
    chunk_id: int
    text: str

class RAGIndex:
    def __init__(self, embedder, docs: List[DocChunk], embs: Optional[np.ndarray]):
        self.embedder = embedder
        self.docs = docs
        self.embs = embs

    def ready(self) -> bool:
        return self.embedder is not None and self.embs is not None and len(self.docs) == self.embs.shape[0]

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Tuple[DocChunk, float]]:
        if not self.ready():
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        sims = (self.embs @ q.T).squeeze(1)
        idx = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.docs[i], float(sims[i])) for i in idx]

_index: Optional[RAGIndex] = None
_facts: Optional[dict] = None

def _paths():
    os.makedirs(INDEX_DIR, exist_ok=True)
    return os.path.join(INDEX_DIR, "embeddings.npy"), os.path.join(INDEX_DIR, "metadata.pkl")

def _save(emb, docs):
    emb_p, meta_p = _paths()
    np.save(emb_p, emb)
    import pickle
    with open(meta_p, "wb") as f:
        pickle.dump(docs, f)

def _load():
    emb_p, meta_p = _paths()
    if not (os.path.exists(emb_p) and os.path.exists(meta_p)):
        return None, None
    emb = np.load(emb_p)
    import pickle
    with open(meta_p, "rb") as f:
        docs = pickle.load(f)
    return emb, docs

def _load_docs() -> Tuple[List[DocChunk], dict]:
    os.makedirs(DATA_DIR, exist_ok=True)
    docs: List[DocChunk] = []
    facts = {}

    # facts.json
    fpath = os.path.join(DATA_DIR, "facts.json")
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                facts = json.load(f)
            # flatten a readable facts text for retrieval
            basics = facts.get("basics", {})
            highlights = facts.get("highlights", [])
            faq = facts.get("faq", {})
            flat = []
            if basics:
                flat.append("BASICS: " + "; ".join([f"{k}: {v}" for k, v in basics.items() if not isinstance(v, dict)]))
            if highlights:
                flat.append("HIGHLIGHTS: " + " | ".join(highlights))
            if faq:
                for k, v in faq.items():
                    flat.append(f"FAQ[{k}]: {v}")
            blob = "\n".join(flat)
            for i, c in enumerate(_chunk(blob, CHUNK_SIZE, CHUNK_OVERLAP)):
                docs.append(DocChunk("facts.json", "Facts", i, c))
        except Exception as e:
            logger.error(f"facts.json load error: {e}")

    # portfolio/*.md (optional)
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "portfolio", "*.md"))):
        title = os.path.splitext(os.path.basename(p))[0]
        for i, c in enumerate(_chunk(_read_text(p), CHUNK_SIZE, CHUNK_OVERLAP)):
            docs.append(DocChunk(f"portfolio/{os.path.basename(p)}", title, i, c))

    # cv.pdf (optional)
    cvp = os.path.join(DATA_DIR, "cv.pdf")
    if os.path.exists(cvp):
        txt = _read_pdf(cvp)
        for i, c in enumerate(_chunk(txt, CHUNK_SIZE, CHUNK_OVERLAP)):
            docs.append(DocChunk("cv.pdf", "CV", i, c))

    logger.info(f"Loaded raw docs: {len(docs)} chunks")
    return docs, facts

def _build_index() -> RAGIndex:
    if SentenceTransformer is None:
        return RAGIndex(None, [], None)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    docs, _ = _load_docs()
    if not docs:
        logger.warning("No docs found under data/. RAG disabled.")
        return RAGIndex(embedder, [], None)
    t0 = time.time()
    embs = embedder.encode([d.text for d in docs], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    logger.info(f"Encoded {len(docs)} chunks in {time.time()-t0:.2f}s")
    _save(embs, docs)
    return RAGIndex(embedder, docs, embs)

def get_index() -> RAGIndex:
    global _index, _facts
    if _index is not None and _index.ready():
        return _index
    emb, docs = _load()
    embedder = SentenceTransformer(EMBEDDING_MODEL) if SentenceTransformer else None
    if emb is not None and docs is not None and embedder is not None:
        _index = RAGIndex(embedder, docs, emb)
        logger.info("RAG index loaded from disk")
    else:
        _index = _build_index()

    # warm facts
    facts_p = os.path.join(DATA_DIR, "facts.json")
    try:
        _facts = json.load(open(facts_p, "r", encoding="utf-8")) if os.path.exists(facts_p) else {}
    except Exception:
        _facts = {}
    return _index

def retrieve(query: str, top_k: Optional[int] = None) -> List[Tuple[DocChunk, float]]:
    idx = get_index()
    k = top_k if top_k and top_k > 0 else TOP_K_DEFAULT
    return idx.retrieve(query, k)

def get_facts() -> dict:
    global _facts
    if _facts is None:
        facts_p = os.path.join(DATA_DIR, "facts.json")
        try:
            _facts = json.load(open(facts_p, "r", encoding="utf-8")) if os.path.exists(facts_p) else {}
        except Exception:
            _facts = {}
    return _facts or {}
