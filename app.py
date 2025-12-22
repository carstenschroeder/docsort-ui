import os
import re
import json
import time
import math
import html
import queue
import shutil
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, BaseLoader

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
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with DB_LOCK:
        conn = db()
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
        conn.close()

def reset_stuck_processing():
    """Reset documents stuck in PROCESSING status back to NEW on startup."""
    with DB_LOCK:
        conn = db()
        conn.execute("""
            UPDATE documents
            SET status='NEW', updated_at=?
            WHERE status='PROCESSING'
        """, (now_iso(),))
        conn.commit()
        conn.close()

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
            "name": d.name if rel != "." else d.name,
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
        conn = db()
        rows = conn.execute("SELECT folder_rel, n, centroid_json FROM folder_profiles").fetchall()
        conn.close()
    out: Dict[str, Tuple[int, List[float]]] = {}
    for r in rows:
        out[r["folder_rel"]] = (int(r["n"]), json.loads(r["centroid_json"]))
    return out

def upsert_folder_profile(folder_rel: str, centroid: List[float], n: int):
    with DB_LOCK:
        conn = db()
        conn.execute("""
            INSERT INTO folder_profiles(folder_rel, n, centroid_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(folder_rel) DO UPDATE SET
                n=excluded.n,
                centroid_json=excluded.centroid_json,
                updated_at=excluded.updated_at
        """, (folder_rel, int(n), json.dumps(centroid), now_iso()))
        conn.commit()
        conn.close()

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
            conn = db()
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
            conn.close()

def pick_one_for_processing() -> Optional[int]:
    with DB_LOCK:
        conn = db()
        row = conn.execute("""
            SELECT id FROM documents
            WHERE status='NEW'
            ORDER BY id ASC
            LIMIT 1
        """).fetchone()
        if not row:
            conn.close()
            return None
        doc_id = int(row["id"])
        conn.execute("UPDATE documents SET status='PROCESSING', updated_at=? WHERE id=?", (now_iso(), doc_id))
        conn.commit()
        conn.close()
    return doc_id

def set_doc_fields(doc_id: int, **fields):
    keys = list(fields.keys())
    if not keys:
        return
    cols = ", ".join([f"{k}=?" for k in keys] + ["updated_at=?"])
    vals = [fields[k] for k in keys] + [now_iso()]
    with DB_LOCK:
        conn = db()
        conn.execute(f"UPDATE documents SET {cols} WHERE id=?", (*vals, doc_id))
        conn.commit()
        conn.close()

def get_doc(doc_id: int) -> sqlite3.Row:
    with DB_LOCK:
        conn = db()
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        conn.close()
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

INDEX_TMPL = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DocSort Review</title>
  <style>
    :root {
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-light: #eef2ff;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --gray-50: #f9fafb;
      --gray-100: #f3f4f6;
      --gray-200: #e5e7eb;
      --gray-300: #d1d5db;
      --gray-400: #9ca3af;
      --gray-500: #6b7280;
      --gray-600: #4b5563;
      --gray-700: #374151;
      --gray-800: #1f2937;
      --gray-900: #111827;
      --radius: 12px;
      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    * { box-sizing: border-box; }
    body { 
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      margin: 0; 
      background: var(--gray-50);
      color: var(--gray-800);
      line-height: 1.5;
    }
    .wrap { 
      display: grid; 
      grid-template-columns: 340px 1fr; 
      height: 100vh; 
    }
    .left { 
      background: white;
      border-right: 1px solid var(--gray-200); 
      overflow: auto; 
      padding: 20px;
      display: flex;
      flex-direction: column;
    }
    .left-header {
      padding-bottom: 16px;
      border-bottom: 1px solid var(--gray-100);
      margin-bottom: 16px;
    }
    .logo {
      font-size: 20px;
      font-weight: 700;
      color: var(--gray-900);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .logo-icon {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 16px;
    }
    .right { 
      overflow: auto; 
      padding: 24px 32px;
    }
    .toolbar { 
      display: flex; 
      gap: 12px; 
      align-items: center; 
      margin-bottom: 20px;
    }
    .muted { color: var(--gray-500); font-size: 13px; }
    .root-path {
      background: var(--gray-100);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--gray-600);
      margin-bottom: 16px;
    }
    .root-path code {
      color: var(--gray-700);
      font-weight: 500;
    }
    .folder-tree {
      flex: 1;
      overflow: auto;
    }
    .folder { 
      padding: 8px 12px; 
      border-radius: 8px;
      transition: all 0.15s ease;
      cursor: pointer;
    }
    .folder:hover { 
      background: var(--gray-100);
    }
    .folder.drop-hover { 
      background: var(--primary-light); 
      outline: 2px dashed var(--primary);
      outline-offset: -2px;
    }
    details { margin-left: 0; }
    details > details { margin-left: 16px; }
    summary { 
      cursor: pointer; 
      list-style: none;
      padding: 4px 0;
    }
    summary::-webkit-details-marker { display: none; }
    summary::before {
      content: '▶';
      display: inline-block;
      width: 16px;
      font-size: 10px;
      color: var(--gray-400);
      transition: transform 0.15s ease;
    }
    details[open] > summary::before {
      transform: rotate(90deg);
    }
    .folder-node { 
      padding: 6px 10px; 
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
    }
    .folder-node b {
      font-weight: 500;
      color: var(--gray-700);
    }
    .folder-icon {
      color: var(--primary);
      font-size: 14px;
    }
    .queue-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .queue-title {
      font-size: 24px;
      font-weight: 700;
      color: var(--gray-900);
    }
    .stats {
      display: flex;
      gap: 16px;
    }
    .stat {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--gray-600);
    }
    .stat-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .stat-dot.ready { background: var(--success); }
    .stat-dot.processing { background: var(--warning); }
    .stat-dot.error { background: var(--error); }
    .stat-dot.moved { background: var(--gray-400); }
    .queue { 
      display: grid; 
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); 
      gap: 16px; 
    }
    .card { 
      background: white;
      border: 1px solid var(--gray-200); 
      border-radius: var(--radius); 
      padding: 16px;
      box-shadow: var(--shadow-sm);
      transition: all 0.2s ease;
    }
    .card:hover {
      box-shadow: var(--shadow-md);
      border-color: var(--gray-300);
    }
    .card[draggable="true"] { cursor: grab; }
    .card:active { cursor: grabbing; }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 12px;
    }
    .card-title {
      font-weight: 600;
      font-size: 14px;
      color: var(--gray-900);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    .tag { 
      font-size: 11px; 
      font-weight: 600;
      padding: 4px 10px; 
      border-radius: 999px; 
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .tag-ready { background: #d1fae5; color: #065f46; }
    .tag-processing { background: #fef3c7; color: #92400e; }
    .tag-error { background: #fee2e2; color: #991b1b; }
    .tag-new { background: var(--primary-light); color: var(--primary); }
    .tag-moved { background: var(--gray-100); color: var(--gray-600); }
    .suggestion-box {
      background: var(--gray-50);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
    }
    .suggestion-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--gray-500);
      margin-bottom: 4px;
    }
    .suggestion-value {
      font-size: 13px;
      color: var(--gray-700);
      font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    }
    .suggestion-score {
      color: var(--gray-400);
      font-size: 12px;
    }
    .card-actions {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .btn { 
      border: 1px solid var(--gray-300); 
      border-radius: 8px; 
      padding: 8px 14px; 
      background: white; 
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      color: var(--gray-700);
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { 
      background: var(--gray-50);
      border-color: var(--gray-400);
    }
    .btn-primary {
      background: var(--primary);
      border-color: var(--primary);
      color: white;
    }
    .btn-primary:hover {
      background: var(--primary-hover);
      border-color: var(--primary-hover);
    }
    .btn-sm {
      padding: 6px 10px;
      font-size: 12px;
    }
    .toast { 
      position: fixed; 
      right: 24px; 
      bottom: 24px; 
      background: var(--gray-900); 
      color: white; 
      padding: 14px 20px; 
      border-radius: var(--radius);
      box-shadow: var(--shadow-lg);
      opacity: 0; 
      transform: translateY(10px);
      transition: all 0.2s ease;
      font-size: 14px;
      font-weight: 500;
    }
    .toast.show { 
      opacity: 1;
      transform: translateY(0);
    }
    a { 
      color: var(--primary); 
      text-decoration: none;
      font-weight: 500;
      font-size: 13px;
    }
    a:hover { text-decoration: underline; }
    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: var(--gray-500);
    }
    .empty-state-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div class="left-header">
      <div class="logo">
        <div class="logo-icon">📄</div>
        DocSort
      </div>
    </div>
    <div class="root-path">
      <span class="muted">Root:</span> <code>{{ root_dir }}</code>
    </div>
    <div class="muted" style="margin-bottom: 12px; font-size: 12px;">
      Drag documents onto folders to sort them
    </div>
    <div class="folder-tree">
      {{ folder_tree_html | safe }}
    </div>
  </div>

  <div class="right">
    <div class="queue-header">
      <div class="queue-title">Document Queue</div>
      <button class="btn btn-primary" onclick="refreshQueue()">
        ↻ Refresh
      </button>
    </div>
    <div class="stats" id="stats"></div>
    <div style="height: 20px"></div>
    <div class="queue" id="queue"></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

function getTagClass(status) {
  const map = {
    'READY': 'tag-ready',
    'PROCESSING': 'tag-processing',
    'ERROR': 'tag-error',
    'NEW': 'tag-new',
    'MOVED': 'tag-moved'
  };
  return map[status] || 'tag-new';
}

async function refreshQueue() {
  const res = await fetch('/api/queue');
  const data = await res.json();
  const el = document.getElementById('queue');
  const statsEl = document.getElementById('stats');
  
  statsEl.innerHTML = `
    <div class="stat"><span class="stat-dot ready"></span>${data.ready} Ready</div>
    <div class="stat"><span class="stat-dot processing"></span>${data.processing} Processing</div>
    <div class="stat"><span class="stat-dot error"></span>${data.error} Error</div>
    <div class="stat"><span class="stat-dot moved"></span>${data.moved} Moved</div>
  `;

  if (data.items.length === 0) {
    el.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <div class="empty-state-icon">📭</div>
        <div>No documents in queue</div>
        <div class="muted" style="margin-top: 8px;">Drop files into the input folder to get started</div>
      </div>
    `;
    return;
  }

  el.innerHTML = '';

  for (const d of data.items) {
    const card = document.createElement('div');
    card.className = 'card';
    card.draggable = true;
    card.dataset.docId = d.id;

    const sug = d.suggestions && d.suggestions.length ? d.suggestions[0] : null;
    const sugFolder = sug ? sug.folder_rel : '—';
    const sugScore = sug ? sug.score : '';

    card.innerHTML = `
      <div class="card-header">
        <div class="card-title" title="${d.name}">${d.name}</div>
        <div class="tag ${getTagClass(d.status)}">${d.status}</div>
      </div>
      <div class="suggestion-box">
        <div class="suggestion-label">Suggested folder</div>
        <div class="suggestion-value">${sugFolder} ${sugScore ? `<span class="suggestion-score">(${sugScore})</span>` : ''}</div>
      </div>
      <div class="card-actions">
        <a href="/doc/${d.id}">View Details</a>
        ${d.status === 'ERROR' ? `<button class="btn btn-sm" onclick="retryDoc(${d.id})">Retry</button>` : ''}
      </div>
    `;

    card.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', String(d.id));
      card.style.opacity = '0.5';
    });
    card.addEventListener('dragend', () => {
      card.style.opacity = '1';
    });

    el.appendChild(card);
  }
}

document.querySelectorAll('[data-folder-rel]').forEach(node => {
  node.addEventListener('dragover', (e) => {
    e.preventDefault();
    node.classList.add('drop-hover');
  });
  node.addEventListener('dragleave', () => node.classList.remove('drop-hover'));
  node.addEventListener('drop', async (e) => {
    e.preventDefault();
    node.classList.remove('drop-hover');

    const docId = e.dataTransfer.getData('text/plain');
    const folderRel = node.dataset.folderRel;

    const res = await fetch(`/api/docs/${docId}/assign`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({folder_rel_path: folderRel})
    });

    if (res.ok) {
      toast(`✓ Moved to ${folderRel}`);
      refreshQueue();
    } else {
      const msg = await res.text();
      alert(msg);
    }
  });
});

async function retryDoc(docId) {
  const res = await fetch(`/api/docs/${docId}/retry`, { method: 'POST' });
  if (res.ok) {
    toast('Retry queued');
    refreshQueue();
  } else {
    alert(await res.text());
  }
}

refreshQueue();
setInterval(refreshQueue, 5000);
</script>
</body>
</html>
"""

DOC_TMPL = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Doc {{ doc.id }} - DocSort</title>
  <style>
    :root {
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-light: #eef2ff;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --gray-50: #f9fafb;
      --gray-100: #f3f4f6;
      --gray-200: #e5e7eb;
      --gray-300: #d1d5db;
      --gray-400: #9ca3af;
      --gray-500: #6b7280;
      --gray-600: #4b5563;
      --gray-700: #374151;
      --gray-800: #1f2937;
      --gray-900: #111827;
      --radius: 12px;
      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    * { box-sizing: border-box; }
    body { 
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      margin: 0; 
      padding: 32px;
      background: var(--gray-50);
      color: var(--gray-800);
      line-height: 1.5;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
    }
    .back-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--gray-600);
      font-size: 14px;
      font-weight: 500;
      text-decoration: none;
      margin-bottom: 16px;
    }
    .back-link:hover {
      color: var(--primary);
    }
    .doc-title {
      font-size: 24px;
      font-weight: 700;
      color: var(--gray-900);
      margin-bottom: 8px;
    }
    .muted { color: var(--gray-500); font-size: 13px; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .meta-item {
      background: var(--gray-100);
      padding: 10px 14px;
      border-radius: 8px;
    }
    .meta-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--gray-500);
      margin-bottom: 4px;
    }
    .meta-value {
      font-size: 13px;
      color: var(--gray-700);
      font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
      word-break: break-all;
    }
    .box { 
      background: white;
      border: 1px solid var(--gray-200); 
      border-radius: var(--radius); 
      padding: 20px;
      margin-top: 20px;
      box-shadow: var(--shadow-sm);
    }
    .box-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--gray-900);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .error-box {
      background: #fef2f2;
      border-color: #fecaca;
    }
    .error-box .box-title {
      color: var(--error);
    }
    pre { 
      white-space: pre-wrap; 
      word-break: break-word; 
      background: var(--gray-900); 
      color: #e5e7eb; 
      padding: 20px; 
      border-radius: var(--radius); 
      overflow: auto;
      font-size: 13px;
      line-height: 1.6;
      margin: 0;
    }
    .btn { 
      border: 1px solid var(--gray-300); 
      border-radius: 8px; 
      padding: 10px 16px; 
      background: white; 
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      color: var(--gray-700);
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { 
      background: var(--gray-50);
      border-color: var(--gray-400);
    }
    .btn-primary {
      background: var(--primary);
      border-color: var(--primary);
      color: white;
    }
    .btn-primary:hover {
      background: var(--primary-hover);
      border-color: var(--primary-hover);
    }
    .btn-error {
      background: var(--error);
      border-color: var(--error);
      color: white;
    }
    .btn-error:hover {
      background: #dc2626;
    }
    a { 
      color: var(--primary); 
      text-decoration: none;
      font-weight: 500;
    }
    a:hover { text-decoration: underline; }
    code { 
      font-size: 12px;
      font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    }
    table { 
      border-collapse: collapse; 
      width: 100%;
    }
    th {
      text-align: left;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--gray-500);
      padding: 12px 16px;
      background: var(--gray-50);
      border-bottom: 1px solid var(--gray-200);
    }
    td { 
      padding: 14px 16px;
      font-size: 14px;
      border-bottom: 1px solid var(--gray-100);
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover {
      background: var(--gray-50);
    }
    .score-badge {
      background: var(--gray-100);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: var(--gray-600);
    }
    .keywords {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .keyword {
      background: var(--primary-light);
      color: var(--primary);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
    }
    .tag { 
      font-size: 11px; 
      font-weight: 600;
      padding: 4px 10px; 
      border-radius: 999px; 
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .tag-ready { background: #d1fae5; color: #065f46; }
    .tag-processing { background: #fef3c7; color: #92400e; }
    .tag-error { background: #fee2e2; color: #991b1b; }
    .tag-new { background: var(--primary-light); color: var(--primary); }
    .tag-moved { background: var(--gray-100); color: var(--gray-600); }
  </style>
</head>
<body>
  <div class="container">
    <a href="/" class="back-link">← Back to Queue</a>
    
    <div class="header">
      <div>
        <div class="doc-title">{{ doc.name }}</div>
        <span class="tag tag-{{ doc.status | lower }}">{{ doc.status }}</span>
      </div>
    </div>

    <div class="meta-grid">
      <div class="meta-item">
        <div class="meta-label">Input Path</div>
        <div class="meta-value">{{ doc.input_path }}</div>
      </div>
      {% if doc.moved_path %}
      <div class="meta-item">
        <div class="meta-label">Moved To</div>
        <div class="meta-value">{{ doc.moved_path }}</div>
      </div>
      {% endif %}
    </div>

    {% if doc.error %}
    <div class="box error-box">
      <div class="box-title">⚠ Error</div>
      <div class="muted">{{ doc.error }}</div>
      <button class="btn btn-error" style="margin-top: 16px" onclick="retry()">Retry Processing</button>
    </div>
    {% endif %}

    <div class="box">
      <div class="box-title">📁 Folder Suggestions</div>
      {% if doc.suggestions %}
        <table>
          <thead>
            <tr><th>#</th><th>Folder</th><th>Score</th><th>Action</th></tr>
          </thead>
          <tbody>
            {% for s in doc.suggestions %}
            <tr>
              <td>{{ loop.index }}</td>
              <td><code>{{ s.folder_rel }}</code></td>
              <td><span class="score-badge">{{ s.score }}</span></td>
              <td><button class="btn btn-primary" onclick="assign('{{ s.folder_rel | e }}')">Move Here</button></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <div class="muted">No suggestions available yet. Folder profiles may need to be generated.</div>
      {% endif %}
    </div>

    <div class="box">
      <div class="box-title">📄 Extracted Content</div>
      <div class="keywords">
        {% for w in keywords %}<span class="keyword">{{ w }}</span>{% endfor %}
      </div>
      <pre>{{ preview_text }}</pre>
    </div>
  </div>

<script>
async function assign(folderRel) {
  const res = await fetch(`/api/docs/{{ doc.id }}/assign`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({folder_rel_path: folderRel})
  });
  if (res.ok) {
    location.href = '/';
  } else {
    alert(await res.text());
  }
}
async function retry() {
  const res = await fetch(`/api/docs/{{ doc.id }}/retry`, {
    method: 'POST'
  });
  if (res.ok) {
    location.reload();
  } else {
    alert(await res.text());
  }
}
</script>
</body>
</html>
"""

env = Environment(loader=BaseLoader(), autoescape=True)

def render_folder_tree_html(tree: Dict[str, Any]) -> str:
    def rec(node: Dict[str, Any], is_root: bool = False) -> str:
        name = html.escape(node["name"])
        rel = node["rel"]
        children = node.get("children", [])

        target_attr = f'data-folder-rel="{html.escape(rel)}"' if rel else ""
        label_inner = f'<span class="folder-node folder" {target_attr}><b>{name}</b></span>' if rel else f'<span class="muted"><b>{name}</b></span>'

        if not children:
            return f'<div class="folder-node folder" {target_attr}><b>{name}</b></div>' if rel else f'<div class="muted"><b>{name}</b></div>'

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

app = FastAPI(title="DocSort Review UI")

STOP = threading.Event()

@app.on_event("startup")
def _startup():

    init_db()
    reset_stuck_processing()

    # Cold-start profiles (folder-name embeddings) -> sofort brauchbare Vorschläge
    seed_profiles_from_folder_names()

    threading.Thread(target=scanner_loop, args=(STOP,), daemon=True).start()
    threading.Thread(target=worker_loop, args=(STOP,), daemon=True).start()

@app.on_event("shutdown")
def _shutdown():
    STOP.set()

@app.get("/", response_class=HTMLResponse)
def index():
    tree = build_tree(cfg.ROOT_DIR)
    folder_tree_html = render_folder_tree_html(tree)
    tmpl = env.from_string(INDEX_TMPL)
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

    tmpl = env.from_string(DOC_TMPL)
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
        conn = db()
        rows = conn.execute("""
            SELECT id, input_path, status, suggestions_json
            FROM documents
            ORDER BY
              CASE status
                WHEN 'READY' THEN 0
                WHEN 'ERROR' THEN 1
                WHEN 'PROCESSING' THEN 2
                WHEN 'NEW' THEN 3
                WHEN 'MOVED' THEN 4
                ELSE 5
              END,
              id DESC
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
        conn.close()

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