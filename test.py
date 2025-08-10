import os
import json
from typing import Any, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header,HTTPException
from fastapi import FastAPI, Body, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi import Depends
from fastapi.security import APIKeyHeader
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import httpx



# ======= Ortam değişkenleri =======
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("COLLECTION", "docs_768")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GENERATE_MODEL = os.getenv("GENERATE_MODEL", "llama3.1:8b")
API_KEY = os.getenv("RAG_API_KEY")  # .env'ye koyabilirsin
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)



# ======= FastAPI (default /docs kapalı, custom docs kullanacağız) =======
app = FastAPI(
    title="Retriever API",
    description="Qdrant + Ollama tabanlı minimal RAG servisi",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yzeka.kku.edu.tr", "https://chat.yzeka.kku.edu.tr"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def check_key(x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    
@app.post("/embed-upsert-batch", tags=["index"], dependencies=[Depends(check_key)])
async def embed_upsert_batch(payload: Dict[str, Any] = Body(...)):
    """
    Body: {"items":[{"id":1,"text":"...","meta":{...}}, ...]}
    """
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return JSONResponse({"error":"items gerekli"}, status_code=400)

    points = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for obj in items:
            t = obj.get("text",""); pid = obj.get("id"); meta = obj.get("meta",{}) or {}
            if not t or pid is None: 
                continue
            r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                  json={"model": EMBED_MODEL, "prompt": t})
            r.raise_for_status()
            vec = r.json()["embedding"]
            points.append({"id": pid, "vector": vec, "payload": {"text": t, **meta}})
        if not points:
            return {"ok": True, "upserted": 0}

        body = {"points": points}
        r2 = await client.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                              headers={"Content-Type":"application/json"},
                              content=json.dumps(body))
        r2.raise_for_status()
    return {"ok": True, "upserted": len(points)}

@app.post("/query-advanced", tags=["search"], dependencies=[Depends(check_key)])
async def query_advanced(payload: Dict[str, Any] = Body(...)):
    """
    Body: {"q":"soru","limit":5,"min_score":0.3,"filter":{"must":[{"key":"source","match":{"value":"deneme"}}]}}
    """
    q = payload.get("q",""); limit = int(payload.get("limit",5))
    min_score = float(payload.get("min_score", 0.0))
    qfilter = payload.get("filter")  # Qdrant filter objesi

    if not q:
        return JSONResponse({"error":"q gerekli"}, status_code=400)

    qvec = await embed_text(q)
    body = {"vector": qvec, "limit": limit}
    if qfilter: body["filter"] = qfilter

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                              headers={"Content-Type":"application/json"},
                              content=json.dumps(body))
        r.raise_for_status()
        results = [h for h in r.json().get("result", []) if h.get("score",0)>=min_score]

        context = "\n\n".join([h["payload"].get("text","") for h in results if "payload" in h])
        prompt = f"Soru: {q}\n\nBağlam:\n{context}\n\nCevap:"
        llm = await client.post(f"{OLLAMA_URL}/api/generate",
                                json={"model": GENERATE_MODEL, "prompt": prompt, "stream": False})
        llm.raise_for_status()
        answer = llm.json().get("response","")
    return {"answer": answer, "matches": results}

@app.post("/delete", tags=["admin"], dependencies=[Depends(check_key)])
async def delete_points(payload: Dict[str, Any] = Body(...)):
    ids = payload.get("ids", [])
    if not ids: return JSONResponse({"error":"ids gerekli"}, status_code=400)
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
                              headers={"Content-Type":"application/json"},
                              content=json.dumps({"points": ids}))
        r.raise_for_status()
    return {"ok": True, "deleted": len(ids)}

@app.post("/collections/recreate", tags=["admin"], dependencies=[Depends(check_key)])
async def recreate_collection():
    # UYARI: koleksiyonu sıfırlar!
    async with httpx.AsyncClient(timeout=60.0) as client:
        await client.delete(f"{QDRANT_URL}/collections/{COLLECTION}")
        r = await client.put(f"{QDRANT_URL}/collections/{COLLECTION}",
                             headers={"Content-Type":"application/json"},
                             content=json.dumps({"vectors":{"size":768,"distance":"Cosine"}}))
        r.raise_for_status()
    return {"ok": True}

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

app.add_middleware(RequestIDMiddleware)

# ======= Yardımcılar =======
async def embed_text(text: str) -> list[float]:
    """Ollama embeddings endpointi ile gömme vektörü üret."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]

async def qdrant_upsert(point_id: Any, vector: list[float], payload: Dict[str, Any]) -> None:
    body = {
        "points": [{
            "id": point_id,
            "vector": vector,
            "payload": payload
        }]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            headers={"Content-Type": "application/json"},
            content=json.dumps(body),
        )
        r.raise_for_status()
        

async def qdrant_search(query_vec: list[float], limit: int = 5) -> list[Dict[str, Any]]:
    body = {"vector": query_vec, "limit": limit}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers={"Content-Type": "application/json"},
            content=json.dumps(body),
        )
        r.raise_for_status()
        return r.json().get("result", [])

async def llm_answer(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GENERATE_MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json().get("response", "")

# ======= Root & Health =======
@app.get("/", include_in_schema=False)
def root():
    # Swagger UI'a yönlendir
    return RedirectResponse(url="/docs", status_code=302)

@app.get("/health", tags=["meta"], dependencies=[Depends(check_key)])
def health():
    return {"ok": True}

# ======= Custom Swagger UI (tema butonlu) =======
@app.get("/docs", include_in_schema=False)
def custom_docs(theme: str | None = Query(default=None)):
    # Swagger UI temel HTML'ini üret (temayı burada vermiyoruz)
    html_resp = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Retriever API - Docs"
    )
    html = html_resp.body.decode("utf-8")

    injected = r"""
    <style>
      /* Dark tema stilleri (body.dark aktifken çalışır) */
      body.dark, body.dark .swagger-ui, body.dark .swagger-ui .wrapper {
        background: #111827 !important; color: #e5e7eb !important;
      }
      body.dark .swagger-ui .topbar, 
      body.dark .swagger-ui .info, 
      body.dark .swagger-ui .model-box, 
      body.dark .swagger-ui .opblock, 
      body.dark .swagger-ui .opblock-summary, 
      body.dark .swagger-ui .opblock-section-header, 
      body.dark .swagger-ui .tab, 
      body.dark .swagger-ui .opblock-body pre, 
      body.dark .swagger-ui textarea, 
      body.dark .swagger-ui input, 
      body.dark .swagger-ui select {
        background: #1f2937 !important; color: #e5e7eb !important; border-color: #374151 !important;
      }
      body.dark .swagger-ui .topbar a, 
      body.dark .swagger-ui .opblock-summary-method, 
      body.dark .swagger-ui .opblock-tag,
      body.dark .swagger-ui .opblock-title {
        color: #e5e7eb !important;
      }
      body.dark .swagger-ui .opblock .opblock-summary-description, 
      body.dark .swagger-ui .markdown p, 
      body.dark .swagger-ui .prop-type, 
      body.dark .swagger-ui .parameter__name {
        color: #d1d5db !important;
      }
      body.dark .swagger-ui .info .title small.version-stamp {
        background: #374151 !important; color: #e5e7eb !important;
      }
      /* Tema butonu */
      #theme-toggle {
        position: fixed; right: 14px; bottom: 14px; z-index: 9999;
        padding: 10px 14px; border-radius: 10px; border: none; cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,.2);
      }
    </style>
    <script>
      (function() {
        const KEY = 'swagger_ui_theme';
        const param = new URLSearchParams(location.search).get('theme');
        let theme = (param === 'light' || param === 'dark') ? param : (localStorage.getItem(KEY) || 'dark');

        function applyTheme() {
          if (theme === 'dark') document.body.classList.add('dark');
          else document.body.classList.remove('dark');
          localStorage.setItem(KEY, theme);
          paintButton();
        }

        const btn = document.createElement('button');
        btn.id = 'theme-toggle';

        function paintButton() {
          if (theme === 'dark') {
            btn.textContent = '🌙 Dark';
            btn.style.background = '#1f2937';
            btn.style.color = '#fff';
          } else {
            btn.textContent = '☀️ Light';
            btn.style.background = '#f3f4f6';
            btn.style.color = '#111827';
          }
        }

        btn.onclick = () => { theme = (theme === 'dark') ? 'light' : 'dark'; applyTheme(); };

        document.addEventListener('DOMContentLoaded', () => {
          document.body.appendChild(btn);
          applyTheme();
        });
      })();
    </script>
    """

    return HTMLResponse(html + injected)

@app.get("/docs/oauth2-redirect", include_in_schema=False)
def swagger_redirect():
    return get_swagger_ui_oauth2_redirect_html()

# ======= API: Embed + Upsert =======
@app.post("/embed-upsert", tags=["index"])
async def embed_upsert(payload: Dict[str, Any] = Body(...)):
    """
    Body:
    {{
      "id": 1,
      "text": "metin",
      "meta": {{"source":"demo"}}
    }}
    """
    text = payload.get("text", "")
    pid = payload.get("id", None)
    meta = payload.get("meta", {}) or {}

    if not text or pid is None:
        return JSONResponse({"error": "id ve text gerekli"}, status_code=400)

    vec = await embed_text(text)
    await qdrant_upsert(pid, vec, {"text": text, **meta})
    return {"ok": True, "id": pid}
    
# ======= API: Query =======
@app.post("/query", tags=["search"])
async def query(payload: Dict[str, Any] = Body(...)):
    """
    Body:
    {{
      "q": "soru", "limit": 5
    }}
    """
    q = payload.get("q", "")
    limit = int(payload.get("limit", 5))
    if not q:
        return JSONResponse({"error": "q gerekli"}, status_code=400)

    qvec = await embed_text(q)
    hits = await qdrant_search(qvec, limit=limit)

    # Basit RAG istemi
    context = "\n\n".join([h.get("payload", {}).get("text", "") for h in hits if h.get("payload")])
    prompt = (
        f"Soru: {q}\n\n"
        f"Bağlam:\n{context}\n\n"
        f"Cevap:"
    )
    answer = await llm_answer(prompt)
    return {"answer": answer, "matches": hits}

from fastapi import UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse

# --- Basit chunk fonksiyonu ---
def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100):
    # yaklaşık karakter tabanlı basit bölücü (hızlı ve yeterli)
    size = max_tokens * 4  # ~ 1 token ≈ 4 char kabulü
    ov   = overlap * 4
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i+size, n)
        chunk = text[i:j]
        out.append(chunk.strip())
        i = j - ov
        if i < 0: i = 0
    return [c for c in out if c]

# --- Basit extractor ---
def extract_text_from_upload(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(data))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    if name.endswith(".docx"):
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    
    if name.endswith(".xlsx"):
        import openpyxl
        from io import BytesIO
        wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
        texts = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
            # None değerleri temizle ve birleştir
                line = " ".join(str(cell) for cell in row if cell is not None)
                if line.strip():
                    texts.append(line.strip())
    return "\n".join(texts)


    raise ValueError("Desteklenmeyen dosya türü (txt, pdf, docx, xlsx deneyin).")

# --- Web: Upload sayfası (GET) ---

@app.get("/upload", include_in_schema=False)
def upload_form() -> HTMLResponse:
    # API key hem Basic Auth var hem de burada form alanı ile de alalım (opsiyonel)
    html = """
    <html>
    <head><title>RAG Upload</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; }
      .card { max-width: 720px; margin:auto; padding:24px; border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.08); }
      .row { margin:12px 0 }
      input[type="file"], input[type="text"], input[type="number"] { width:100%; padding:10px; }
      button { padding:10px 16px; border-radius:10px; border:none; background:#111827; color:#fff; cursor:pointer; }
      .hint { color:#6b7280; font-size:12px; }
      .ok { color: #065f46; }
      .err { color: #991b1b; }
    </style></head>
    <body>
      <div class="card">
        <h2>Dosya Yükle & İndeksle</h2>
        <form id="f" method="post" action="/upload" enctype="multipart/form-data">
          <div class="row"><input type="file" name="file" accept=".txt,.pdf,.docx" required /></div>
          <div class="row"><input type="text" name="source" placeholder="source etiketi (örn: deneme)" /></div>
          <div class="row">
            <label>Chunk boyutu (yaklaşık token):</label>
            <input type="number" name="max_tokens" value="800" min="200" max="2000"/>
          </div>
          <div class="row">
            <label>Overlap (yaklaşık token):</label>
            <input type="number" name="overlap" value="100" min="0" max="400"/>
          </div>
          <div class="row">
            <label>API Key (X-API-KEY) – boş bırakabilirsiniz:</label>
            <input type="text" name="api_key" placeholder="opsiyonel"/>
            <div class="hint">Caddy Basic Auth zaten var. Uç ayrıca API key de kontrol edebilir.</div>
          </div>
          <button type="submit">Yükle & İndeksle</button>
        </form>
        <div id="out" class="row"></div>
        <script>
          // X-API-KEY'i header olarak göndermek için JS ile submit edelim
          const form = document.getElementById('f');
          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = new FormData(form);
            const apiKey = data.get('api_key');
            data.delete('api_key');
            const r = await fetch('/upload', {
              method: 'POST',
              body: data,
              headers: apiKey ? {'X-API-KEY': apiKey} : {}
            });
            const out = document.getElementById('out');
            try {
              const j = await r.json();
              out.innerHTML = r.ok ? '<div class="ok">✅ '+JSON.stringify(j)+'</div>'
                                   : '<div class="err">❌ '+JSON.stringify(j)+'</div>';
            } catch(e) {
              out.innerHTML = '<div class="err">❌ '+r.status+' '+r.statusText+'</div>';
            }
          });
        </script>
      </div>
    </body></html>
    """
    return HTMLResponse(html)

# --- Web: Upload işleme (POST) ---
@app.post("/upload", tags=["index"])
async def upload_file(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    max_tokens: int = Form(default=800),
    overlap: int = Form(default=100),
    x_api_key: str | None = Depends(lambda x_api_key=Header(default=None, alias="X-API-KEY"): x_api_key)
):
    # API key kontrol (opsiyonel): .env'de RAG_API_KEY varsa zorla
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    raw = await file.read()
    try:
        text = extract_text_from_upload(file.filename, raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Boş içerik")
    logger = logging.getLogger("uvicorn")
    logger.info(f"upload ok: file={file.filename}, chunks={len(chunks)} source={source or 'upload'}")

    # embed + batch upsert
    import os
    MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "80"))
    points = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        # embed modelini bir kez dene (bağlantı)
        for idx, ch in enumerate(chunks, start=1):
            r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                                  json={"model": EMBED_MODEL, "prompt": ch})
            r.raise_for_status()
            vec = r.json()["embedding"]
            points.append({
                "id": f"{file.filename}-{idx}",
                "vector": vec,
                "payload": {"text": ch, "source": source or "upload", "filename": file.filename, "chunk": idx}
            })
        body = {"points": points}
        r2 = await client.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                              headers={"Content-Type":"application/json"},
                              content=json.dumps(body))
        r2.raise_for_status()
    chunks = chunks[:MAX_CHUNKS]
    return {"ok": True, "filename": file.filename, "chunks": len(chunks)}



