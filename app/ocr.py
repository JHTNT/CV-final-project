from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None
    bbox: list[list[float]] | None


@dataclass(frozen=True)
class OCRResult:
    full_text: str
    lines: list[OCRLine]


@lru_cache(maxsize=1)
def _get_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=os.getenv("OCR_LANG", "ch"),
    )


def run_ocr(pil_image) -> OCRResult:
    ocr = _get_ocr()
    image = np.array(pil_image)
    raw: Any = ocr.predict(image)

    lines: list[OCRLine] = []
    texts: list[str] = []

    if os.getenv("OCR_DEBUG", "False").lower() == "true":
        try:
            raw_type = type(raw).__name__
            raw_len = len(raw) if hasattr(raw, "__len__") else None
            logger.info(
                "OCR raw type=%s len=%s image_shape=%s dtype=%s",
                raw_type,
                raw_len,
                getattr(image, "shape", None),
                getattr(image, "dtype", None),
            )
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                logger.info("OCR raw[0] keys=%s", sorted(list(raw[0].keys())))
        except Exception:
            logger.exception("Failed to log OCR debug info")

    def _to_py_scalar(v: Any):
        # Convert numpy scalar types to Python scalars
        if isinstance(v, np.generic):
            return v.item()
        return v

    def _normalize_bbox(bbox: Any) -> list[list[float]] | None:
        if bbox is None:
            return None

        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        # Some outputs may provide [x1, y1, x2, y2]
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(_to_py_scalar(x), (int, float)) for x in bbox)
        ):
            x1, y1, x2, y2 = bbox
            return [
                [float(_to_py_scalar(x1)), float(_to_py_scalar(y1))],
                [float(_to_py_scalar(x2)), float(_to_py_scalar(y1))],
                [float(_to_py_scalar(x2)), float(_to_py_scalar(y2))],
                [float(_to_py_scalar(x1)), float(_to_py_scalar(y2))],
            ]

        # Expected polygon: [[x,y], [x,y], [x,y], [x,y]]
        if isinstance(bbox, (list, tuple)):
            pts: list[list[float]] = []
            for p in bbox:
                if isinstance(p, np.ndarray):
                    p = p.tolist()
                if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                    return None
                x = _to_py_scalar(p[0])
                y = _to_py_scalar(p[1])
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    return None
                pts.append([float(x), float(y)])
            return pts

        return None

    def add_line(*, text: str, confidence: float | None = None, bbox: Any = None):
        text = (text or "").strip()
        if not text:
            return
        conf = float(_to_py_scalar(confidence)) if confidence is not None else None
        norm_bbox = _normalize_bbox(bbox)
        lines.append(OCRLine(text=text, confidence=conf, bbox=norm_bbox))
        texts.append(text)

    def parse_predict_output(obj: Any):
        # Newer PaddleOCR (PaddleX pipeline style): list[dict] with rec_texts/rec_scores/rec_polys
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for page in obj:

                def as_list(v: Any) -> list[Any]:
                    if v is None:
                        return []
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                    if isinstance(v, tuple):
                        return list(v)
                    return v if isinstance(v, list) else [v]

                rec_texts = as_list(page.get("rec_texts"))
                rec_scores = as_list(page.get("rec_scores"))
                rec_polys = as_list(page.get("rec_polys"))
                rec_boxes = as_list(page.get("rec_boxes"))

                for i, t in enumerate(rec_texts):
                    conf = rec_scores[i] if i < len(rec_scores) else None
                    bbox = None
                    if i < len(rec_polys):
                        bbox = rec_polys[i]
                    elif i < len(rec_boxes):
                        b = rec_boxes[i]
                        # rec_boxes is typically [x1, y1, x2, y2]
                        if isinstance(b, (list, tuple)) and len(b) == 4:
                            x1, y1, x2, y2 = b
                            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    add_line(text=str(t), confidence=conf, bbox=bbox)
            return True
        return False

    def parse_legacy_output(obj: Any):
        # Legacy PaddleOCR output: nested list with per line [bbox, (text, conf)]
        def iter_items(x: Any):
            if x is None:
                return
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
                yield from iter_items(x[0])
                return
            if isinstance(x, list):
                for it in x:
                    if isinstance(it, list) and len(it) >= 2 and isinstance(it[0], list):
                        yield it

        for item in iter_items(obj):
            bbox = item[0]
            text = None
            conf = None
            if isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                text = item[1][0]
                if len(item[1]) >= 2:
                    conf = item[1][1]
            if text:
                add_line(text=str(text), confidence=conf, bbox=bbox)
        return True

    # Prefer new output format; fall back to legacy.
    if not parse_predict_output(raw):
        parse_legacy_output(raw)

    full_text = "\n".join(texts).strip()
    return OCRResult(full_text=full_text, lines=lines)
