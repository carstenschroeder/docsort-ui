import os
import re
import json
import time
import math
import html
import shutil
import sqlite3
import threading
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

# =========================
# Config
# =========================

@dataclass
class Config:
    ROOT_DIR: Path = Path(os.environ.get("ROOT_DIR", "/data/sorted"))
    INPUT_DIR: Path = Path(os.environ.get("INPUT_DIR", "/data/inbox"))
    STATE_DIR: Path = Path(os.environ.get("STATE_DIR", "/data/state_docsort"))

    DOCLING_URL: str = os.environ.get("DOCLING_URL", "http://127.0.0.1:5001")
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")

    # polling / processing
    SCAN_INTERVAL_SEC: float = float(os.environ.get("SCAN_INTERVAL_SEC", "2.0"))
    FILE_STABLE_SEC: float = float(os.environ.get("FILE_STABLE_SEC", "2.0"))

    # docling params (Form fields laut docs)
    DO_OCR: bool = os.environ.get("DO_OCR", "true").lower() == "true"
    OCR_ENGINE: str = os.environ.get("OCR_ENGINE", "easyocr")
    OCR_LANGS: List[str] = field(default_factory=lambda: [s.strip() for s in os.environ.get("OCR_LANGS", "de,en").split(",") if s.strip()])
    PDF_BACKEND: str = os.environ.get("PDF_BACKEND", "dlparse_v4")
    TO_FORMATS: List[str] = field(default_factory=lambda: [s.strip() for s in os.environ.get("TO_FORMATS", "md,text").split(",") if s.strip()])

    # embedding input size guard
    MAX_EMBED_CHARS: int = int(os.environ.get("MAX_EMBED_CHARS", "30000"))

    # suggestions
    TOPK: int = int(os.environ.get("TOPK", "5"))

    DOCLING_TIMEOUT: float = float(os.environ.get("DOCLING_TIMEOUT", "600.0"))

cfg = Config()

# Validate required directories exist
if not cfg.ROOT_DIR.exists():
    raise SystemExit(f"ERROR: ROOT_DIR does not exist: {cfg.ROOT_DIR}")
if not cfg.INPUT_DIR.exists():
    raise SystemExit(f"ERROR: INPUT_DIR does not exist: {cfg.INPUT_DIR}")

# =========================
# Paths / init
# =========================

cfg.STATE_DIR.mkdir(parents=True, exist_ok=True)
(cfg.STATE_DIR / "extracted").mkdir(parents=True, exist_ok=True)

DB_PATH = cfg.STATE_DIR / "state.sqlite3"
DB_LOCK = threading.Lock()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

@contextmanager
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with DB_LOCK:
        with db() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path TEXT NOT NULL,
                sha256 TEXT,
                status TEXT NOT NULL,
                extracted_md_path TEXT,
                embedding_json TEXT,
                suggestions_json TEXT,
                suggested_folder_rel TEXT,
                chosen_folder_rel TEXT,
                moved_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_input_path ON documents(input_path)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS folder_profiles (
                folder_rel TEXT PRIMARY KEY,
                n INTEGER NOT NULL,
                centroid_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            conn.commit()

def reset_stuck_processing():
    """Reset documents stuck in PROCESSING status back to NEW on startup."""
    with DB_LOCK:
        with db() as conn:
            conn.execute("""
                UPDATE documents
                SET status='NEW', updated_at=?
                WHERE status='PROCESSING'
            """, (now_iso(),))
            conn.commit()

# =========================
# Folder tree utilities
# =========================

def safe_relpath(p: Path, base: Path) -> str:
    return str(p.resolve().relative_to(base.resolve()))

def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def list_all_target_folders(root_dir: Path) -> List[str]:
    root_dir = root_dir.resolve()
    out = []
    for d in root_dir.rglob("*"):
        if d.is_dir() and not any(part.startswith(".") for part in d.relative_to(root_dir).parts):
            rel = safe_relpath(d, root_dir)
            out.append(rel)
    out.sort()
    return out

def build_tree(root_dir: Path) -> Dict[str, Any]:
    root_dir = root_dir.resolve()

    def node_for_dir(d: Path) -> Dict[str, Any]:
        rel = safe_relpath(d, root_dir)
        children = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")]
        children.sort(key=lambda x: x.name.lower())
        return {
            "name": d.name,
            "rel": "" if rel == "." else rel,
            "children": [node_for_dir(c) for c in children],
        }

    return node_for_dir(root_dir)

# =========================
# Text / embeddings / similarity
# =========================

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]{3,}")

def top_keywords(text: str, k: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        counts[w] = counts.get(w, 0) + 1
    # drop ultra-common german filler-ish words (tiny list)
    stop = {"und","der","die","das","ein","eine","mit","für","von","auf","ist","im","in","am","an","zu","den","des","dem"}
    items = [(w,c) for w,c in counts.items() if w not in stop]
    items.sort(key=lambda t: (-t[1], t[0]))
    return [w for w,_ in items[:k]]

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def ollama_embed(text: str) -> List[float]:
    text = text[: cfg.MAX_EMBED_CHARS]
    url = cfg.OLLAMA_URL.rstrip("/") + "/v1/embeddings"
    payload = {"model": cfg.EMBED_MODEL, "input": text}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    emb = data["data"][0]["embedding"]
    return emb

# =========================
# docling-serve extraction
# =========================

def docling_extract_md_and_text(file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    """
    Uses docling-serve file endpoint:
    POST /v1/convert/file (multipart/form-data)
    Response format contains document.md_content / text_content (when requested),
    see usage.md. [docling-serve usage.md]
    https://raw.githubusercontent.com/docling-project/docling-serve/main/docs/usage.md
    """
    url = cfg.DOCLING_URL.rstrip("/") + "/v1/convert/file"

    # Form fields are "flat" fields; repeating allowed (to_formats=md & to_formats=text)
    data_fields: Dict[str, Any] = {
        "do_ocr": "true" if cfg.DO_OCR else "false",
        "force_ocr": "false",
        "ocr_engine": cfg.OCR_ENGINE,
        "pdf_backend": cfg.PDF_BACKEND,
        "image_export_mode": "placeholder",
    }
    
    with httpx.Client(timeout=cfg.DOCLING_TIMEOUT) as client:
        with open(file_path, "rb") as f:
            # Build multipart files list with all form fields
            files_list = [
                ("files", (file_path.name, f, "application/octet-stream")),
            ]
            # Add repeated fields as separate file tuples (httpx multipart format)
            for tf in cfg.TO_FORMATS:
                files_list.append(("to_formats", (None, tf)))
            for lang in cfg.OCR_LANGS:
                files_list.append(("ocr_lang", (None, lang)))
            
            r = client.post(url, data=data_fields, files=files_list, headers={"accept": "application/json"})
            r.raise_for_status()
            payload = r.json()

    doc = payload.get("document", {}) or {}
    md = doc.get("md_content", "") or ""
    txt = doc.get("text_content", "") or ""
    return md, txt, payload

# =========================
# Folder profiles (centroids)
# =========================

def load_folder_profiles() -> Dict[str, Tuple[int, List[float]]]:
    with DB_LOCK:
        with db() as conn:
            rows = conn.execute("SELECT folder_rel, n, centroid_json FROM folder_profiles").fetchall()
    out: Dict[str, Tuple[int, List[float]]] = {}
    for r in rows:
        out[r["folder_rel"]] = (int(r["n"]), json.loads(r["centroid_json"]))
    return out

def upsert_folder_profile(folder_rel: str, centroid: List[float], n: int):
    with DB_LOCK:
        with db() as conn:
            conn.execute("""
                INSERT INTO folder_profiles(folder_rel, n, centroid_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(folder_rel) DO UPDATE SET
                    n=excluded.n,
                    centroid_json=excluded.centroid_json,
                    updated_at=excluded.updated_at
            """, (folder_rel, int(n), json.dumps(centroid), now_iso()))
            conn.commit()

def seed_profiles_from_folder_names():
    """
    Cold-start: erzeugt pro Ordner ein Seed-Centroid nur aus Ordnerpfad + Beispiel-Dateinamen.
    Vorteil: sofort Vorschläge möglich, ohne erst tausende Dokumente zu verarbeiten.
    """
    folders = list_all_target_folders(cfg.ROOT_DIR)
    existing = load_folder_profiles()

    for rel in folders:
        if rel in existing:
            continue
        abs_dir = (cfg.ROOT_DIR / rel).resolve()
        # sample file basenames (no content processing!)
        examples = []
        try:
            for p in abs_dir.iterdir():
                if p.is_file():
                    examples.append(p.name)
                if len(examples) >= 25:
                    break
        except Exception:
            pass

        profile_text = f"Folder path: {rel}\n" + ("Examples:\n" + "\n".join(examples) if examples else "")
        try:
            emb = ollama_embed(profile_text)
        except Exception as e:
            # if ollama not ready, skip; will be retried later by manual restart
            print(f"[seed_profiles] failed for {rel}: {e}")
            continue
        # store as n=1 seed (real docs will quickly dominate as n grows)
        upsert_folder_profile(rel, emb, n=1)
        print(f"[seed_profiles] seeded {rel}")

def update_centroid(old_centroid: List[float], old_n: int, new_emb: List[float]) -> Tuple[List[float], int]:
    if not old_centroid or old_n <= 0:
        return new_emb, 1
    n = old_n + 1
    c = [(old_n * oc + ne) / n for oc, ne in zip(old_centroid, new_emb)]
    return c, n

def suggest_folders(doc_emb: List[float], topk: int) -> List[Dict[str, Any]]:
    profiles = load_folder_profiles()
    scored: List[Tuple[float, str]] = []
    for folder_rel, (n, centroid) in profiles.items():
        s = cosine(doc_emb, centroid)
        scored.append((s, folder_rel))
    scored.sort(reverse=True, key=lambda t: t[0])
    out = []
    for s, rel in scored[:topk]:
        out.append({"folder_rel": rel, "score": round(float(s), 4)})
    return out

# =========================
# Document ingestion / processing
# =========================

def file_stable(p: Path, stable_sec: float) -> bool:
    try:
        s1 = p.stat().st_size
        t1 = time.time()
        time.sleep(stable_sec)
        s2 = p.stat().st_size
        t2 = time.time()
        return s1 == s2 and (t2 - t1) >= stable_sec
    except FileNotFoundError:
        return False

def enqueue_new_files():
    for p in cfg.INPUT_DIR.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if not file_stable(p, cfg.FILE_STABLE_SEC):
            continue

        with DB_LOCK:
            with db() as conn:
                row = conn.execute("SELECT id, status FROM documents WHERE input_path=?", (str(p),)).fetchone()
                if row is None:
                    conn.execute("""
                        INSERT INTO documents(input_path, status, created_at, updated_at)
                        VALUES(?, 'NEW', ?, ?)
                    """, (str(p), now_iso(), now_iso()))
                elif row["status"] == "MOVED":
                    conn.execute("""
                        UPDATE documents SET status='NEW', moved_path=NULL, chosen_folder_rel=NULL, updated_at=?
                        WHERE id=?
                    """, (now_iso(), row["id"]))
                conn.commit()

def pick_one_for_processing() -> Optional[int]:
    with DB_LOCK:
        with db() as conn:
            row = conn.execute("""
                SELECT id FROM documents
                WHERE status='NEW'
                ORDER BY id ASC
                LIMIT 1
            """).fetchone()
            if not row:
                return None
            doc_id = int(row["id"])
            conn.execute("UPDATE documents SET status='PROCESSING', updated_at=? WHERE id=?", (now_iso(), doc_id))
            conn.commit()
    return doc_id

ALLOWED_DOC_COLUMNS = {"status", "extracted_md_path", "embedding_json", "suggestions_json",
                        "suggested_folder_rel", "chosen_folder_rel", "moved_path", "error", "sha256"}

def set_doc_fields(doc_id: int, **fields):
    keys = list(fields.keys())
    if not keys:
        return
    for k in keys:
        if k not in ALLOWED_DOC_COLUMNS:
            raise ValueError(f"Invalid column name: {k}")
    cols = ", ".join([f"{k}=?" for k in keys] + ["updated_at=?"])
    vals = [fields[k] for k in keys] + [now_iso()]
    with DB_LOCK:
        with db() as conn:
            conn.execute(f"UPDATE documents SET {cols} WHERE id=?", (*vals, doc_id))
            conn.commit()

def get_doc(doc_id: int) -> sqlite3.Row:
    with DB_LOCK:
        with db() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "doc not found")
    return row

def process_doc(doc_id: int):
    row = get_doc(doc_id)
    input_path = Path(row["input_path"])
    if not input_path.exists():
        set_doc_fields(doc_id, status="ERROR", error="Input file not found anymore")
        return

    try:
        md, txt, raw = docling_extract_md_and_text(input_path)
        extracted_path = cfg.STATE_DIR / "extracted" / f"{doc_id}.md"
        extracted_path.write_text(md or txt or "", encoding="utf-8", errors="ignore")

        # Build embedding input from filename + extracted content
        base = input_path.name
        content = md if md.strip() else txt
        emb_input = f"Filename: {base}\n\n{content}"
        emb = ollama_embed(emb_input)

        suggestions = suggest_folders(emb, cfg.TOPK)
        suggested = suggestions[0]["folder_rel"] if suggestions else None

        set_doc_fields(
            doc_id,
            status="READY",
            extracted_md_path=str(extracted_path),
            embedding_json=json.dumps(emb),
            suggestions_json=json.dumps(suggestions),
            suggested_folder_rel=suggested,
            error=None
        )
    except Exception as e:
        set_doc_fields(doc_id, status="ERROR", error=str(e))

def scanner_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            enqueue_new_files()
        except Exception as e:
            print(f"[scanner] {e}")
        time.sleep(cfg.SCAN_INTERVAL_SEC)

def worker_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        doc_id = pick_one_for_processing()
        if doc_id is None:
            time.sleep(0.5)
            continue
        process_doc(doc_id)

# =========================
# Move / assign
# =========================

def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 1000):
        c = dest_dir / f"{stem} ({i}){suffix}"
        if not c.exists():
            return c
    raise RuntimeError("Too many name collisions")

def assign_doc_to_folder(doc_id: int, folder_rel: str):
    row = get_doc(doc_id)
    if row["status"] not in ("READY", "ERROR", "PROCESSING", "NEW"):
        raise HTTPException(409, f"Cannot assign doc in status {row['status']}")

    # validate folder exists under ROOT_DIR
    target_dir = (cfg.ROOT_DIR / folder_rel).resolve()
    if not is_under_root(target_dir, cfg.ROOT_DIR) or not target_dir.is_dir():
        raise HTTPException(400, "Invalid target folder")

    src = Path(row["input_path"])
    if not src.exists():
        raise HTTPException(410, "Source file missing")

    dest = unique_dest_path(target_dir, src.name)
    shutil.move(str(src), str(dest))

    # update profile with this confirmed example
    emb_json = row["embedding_json"]
    if emb_json:
        emb = json.loads(emb_json)
        profiles = load_folder_profiles()
        old_n, old_c = profiles.get(folder_rel, (0, []))
        new_c, new_n = update_centroid(old_c, old_n, emb)
        upsert_folder_profile(folder_rel, new_c, new_n)

    set_doc_fields(
        doc_id,
        status="MOVED",
        chosen_folder_rel=folder_rel,
        moved_path=str(dest)
    )

# =========================
# Web UI
# =========================

env = Environment(loader=FileSystemLoader('templates'), autoescape=True)

def render_folder_tree_html(tree: Dict[str, Any]) -> str:
    def rec(node: Dict[str, Any], is_root: bool = False) -> str:
        name = html.escape(node["name"])
        rel = node["rel"]
        children = node.get("children", [])

        target_attr = f'data-folder-rel="{html.escape(rel)}"' if rel else ""
        label_inner = f'<span class="folder-node folder" {target_attr}><b>{name}</b></span>' if rel else f'<span class="muted"><b>{name}</b></span>'

        if not children:
            return f'<div class="folder-node folder" style="display: block; margin-left: 16px" {target_attr}><b>{name}</b></div>' if rel else f'<div class="muted" style="display: block"><b>{name}</b></div>'

        inner = "\n".join(rec(c) for c in children)
        open_attr = "open" if is_root else ""
        return f"""
        <details {open_attr}>
          <summary style="display: list-item">{label_inner}</summary>
          <div style="margin-left: 10px">{inner}</div>
        </details>
        """

    return rec(tree, is_root=True)

# =========================
# FastAPI app
# =========================

STOP = threading.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    reset_stuck_processing()

    # Cold-start profiles (folder-name embeddings) -> sofort brauchbare Vorschläge
    seed_profiles_from_folder_names()

    threading.Thread(target=scanner_loop, args=(STOP,), daemon=True).start()
    threading.Thread(target=worker_loop, args=(STOP,), daemon=True).start()

    yield

    STOP.set()

app = FastAPI(title="DocSort Review UI", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    tree = build_tree(cfg.ROOT_DIR)
    folder_tree_html = render_folder_tree_html(tree)
    tmpl = env.get_template('index.html')
    html_out = tmpl.render(folder_tree_html=folder_tree_html, root_dir=str(cfg.ROOT_DIR))
    return HTMLResponse(html_out)

@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def doc_detail(doc_id: int):
    r = get_doc(doc_id)
    suggestions = json.loads(r["suggestions_json"]) if r["suggestions_json"] else []
    extracted = ""
    if r["extracted_md_path"] and Path(r["extracted_md_path"]).exists():
        extracted = Path(r["extracted_md_path"]).read_text(encoding="utf-8", errors="ignore")
    extracted = extracted.strip()

    keywords = top_keywords(extracted, k=12) if extracted else []
    preview_text = extracted[:120000] if extracted else "(no extracted text yet)"

    tmpl = env.get_template('doc.html')
    html_out = tmpl.render(
        doc={
            "id": int(r["id"]),
            "name": Path(r["input_path"]).name,
            "status": r["status"],
            "input_path": r["input_path"],
            "moved_path": r["moved_path"],
            "error": r["error"],
            "suggestions": suggestions,
        },
        preview_text=preview_text,
        keywords=keywords,
        topk=cfg.TOPK
    )
    return HTMLResponse(html_out)

@app.get("/api/queue")
def api_queue():
    with DB_LOCK:
        with db() as conn:
            rows = conn.execute("""
                SELECT id, input_path, status, suggestions_json
                FROM documents
                ORDER BY input_path ASC
                LIMIT 200
            """).fetchall()

            counts = conn.execute("""
                SELECT
                  SUM(CASE WHEN status='READY' THEN 1 ELSE 0 END) as ready,
                  SUM(CASE WHEN status='PROCESSING' THEN 1 ELSE 0 END) as processing,
                  SUM(CASE WHEN status='ERROR' THEN 1 ELSE 0 END) as error,
                  SUM(CASE WHEN status='MOVED' THEN 1 ELSE 0 END) as moved
                FROM documents
            """).fetchone()

    items = []
    for r in rows:
        items.append({
            "id": int(r["id"]),
            "name": Path(r["input_path"]).name,
            "status": r["status"],
            "suggestions": json.loads(r["suggestions_json"]) if r["suggestions_json"] else []
        })

    return JSONResponse({
        "items": items,
        "ready": int(counts["ready"] or 0),
        "processing": int(counts["processing"] or 0),
        "error": int(counts["error"] or 0),
        "moved": int(counts["moved"] or 0),
    })

@app.post("/api/docs/{doc_id}/assign")
def api_assign(doc_id: int, payload: Dict[str, Any]):
    folder_rel = payload.get("folder_rel_path")
    if not isinstance(folder_rel, str) or folder_rel.strip() == "":
        raise HTTPException(400, "folder_rel_path required")
    folder_rel = folder_rel.strip()
    assign_doc_to_folder(doc_id, folder_rel)
    return JSONResponse({"ok": True})

@app.post("/api/docs/{doc_id}/retry")
def api_retry(doc_id: int):
    row = get_doc(doc_id)
    if row["status"] != "ERROR":
        raise HTTPException(409, f"Can only retry docs with ERROR status, got {row['status']}")
    input_path = Path(row["input_path"])
    if not input_path.exists():
        raise HTTPException(410, "Source file missing")
    set_doc_fields(doc_id, status="NEW", error=None)
    return JSONResponse({"ok": True})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)