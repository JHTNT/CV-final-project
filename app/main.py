from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .llm import analyze_text_with_llm
from .ocr import _get_ocr, run_ocr
from .utils.images import load_and_downscale, sniff_mime

load_dotenv(".env", override=True)

app = FastAPI(title="Food Ingredient OCR Demo")
_get_ocr()  # preload OCR model


@lru_cache(maxsize=1)
def _load_index_html_template() -> str:
    path = Path(__file__).resolve().parent / "templates" / "index.html"
    return path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
    max_upload_mb = max(1, max_upload_bytes // (1024 * 1024))


    html = _load_index_html_template()
    return html.replace("__MAX_UPLOAD_MB__", str(max_upload_mb))


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
