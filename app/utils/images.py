from __future__ import annotations

import imghdr
from typing import Literal

from PIL import Image, ImageOps

AllowedMime = Literal["image/jpeg", "image/png", "image/webp"]


def sniff_mime(data: bytes) -> AllowedMime | None:
    kind = imghdr.what(None, h=data)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    return None


def load_and_downscale(image_bytes: bytes, *, max_side: int = 2000) -> Image.Image:
    img = Image.open(io_bytes(image_bytes))
    # iPhone (and many cameras) store orientation in EXIF instead of rotating pixels.
    # If we ignore this, OCR sees sideways/upside-down text and often returns garbage.
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # If EXIF is missing/corrupted, keep original pixels.
        pass
    img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        # Default PIL resize is nearest-neighbor; that's very harmful for OCR.
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            resample = Image.LANCZOS  # type: ignore[attr-defined]
        img = img.resize(new_size, resample=resample)
    return img


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)
