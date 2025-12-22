from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """你是食品標示與食品添加物的助手。使用者會提供一段 OCR 出來的食品成分/營養標示文字。
請做：
1) 盡可能解析出『成分』清單（依標示順序）。
2) 標出可能的過敏原（例如：含麩質穀物、牛奶、蛋、花生、堅果、大豆、芝麻、魚、甲殼類等）
3) 若看到食品添加物/代號（例如：防腐劑、甜味劑、色素、香料、乳化劑等），請列出並簡短說明用途。
4) 若文字不足或不確定，請明確標註不確定性。

輸出請一律用 JSON，欄位：
{
  "ingredients": ["..."],
  "allergens": ["..."],
  "additives": [{"name": "...", "purpose": "..."}],
  "notes": ["..."]
}
只輸出 JSON，不要加任何額外文字。
"""


def analyze_text_with_llm(text: str) -> tuple[dict[str, Any] | None, str]:
    if not os.getenv("OPENAI_API_KEY"):
        return None, "缺少 OPENAI_API_KEY"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return None, "LLM 回傳空內容"

    try:
        return json.loads(content), content
    except Exception:
        # demo: return raw if not valid JSON
        return None, content
