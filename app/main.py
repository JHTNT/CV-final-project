from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .llm import analyze_text_with_llm
from .ocr import run_ocr
from .utils.images import load_and_downscale, sniff_mime

load_dotenv(".env", override=True)

app = FastAPI(title="Food Ingredient OCR Demo")


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Food Ingredient OCR Demo</title>
</head>
<body>
  <h2>食品成分 OCR → LLM 分析（Demo）</h2>
  <form action=\"/analyze\" method=\"post\" enctype=\"multipart/form-data\">
    <input type=\"file\" name=\"image\" accept=\"image/*\" required />
    <button type=\"submit\">上傳並分析</button>
  </form>
  <p>API: POST /analyze (multipart field: image)</p>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()

    if len(data) > int(os.getenv("MAX_UPLOAD_BYTES", "10485760")):
        raise HTTPException(status_code=413, detail="檔案太大")

    mime = sniff_mime(data)
    if mime is None:
        raise HTTPException(status_code=415, detail="只接受 jpg/png/webp")

    pil = load_and_downscale(data)
    ocr_res = run_ocr(pil)

    llm_json, llm_raw = analyze_text_with_llm(ocr_res.full_text)

    return {
        "ocr": {
            "full_text": ocr_res.full_text,
            "lines": [
                {"text": ln.text, "confidence": ln.confidence, "bbox": ln.bbox}
                for ln in ocr_res.lines
            ],
        },
        "llm": {"json": llm_json, "raw": llm_raw},
        "meta": {"mime": mime},
    }
