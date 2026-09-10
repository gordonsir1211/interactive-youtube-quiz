#!/usr/bin/env python3
"""server.py — AI question generation backend for the YouTube quiz page. Runs on port 8000."""
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic()


class GenerateRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=20000)
    count: int = Field(3, ge=1, le=10)
    video_title: str = ""


SYSTEM_PROMPT = """你是一位教學設計助理，負責依據老師提供的 YouTube 影片文字稿或內容重點，
設計適合在觀看影片「之前」讓學生作答的四選一多項選擇題，用來檢查學生是否已預先閱讀/理解重點，
或作為觀看前的暖身思考題。

規則：
- 每一題必須有且只有一個正確答案，四個選項中只有一個正確。
- 題目與選項一律使用繁體中文（除非文字稿本身是其他語言，此時可保留專有名詞原文）。
- 題目應涵蓋文字稿中不同的重點，避免重複問同一件事。
- 選項長度接近、避免明顯過長或過短的正確答案洩漏答案。
- 干擾選項（錯誤選項）要合理、相關，不能明顯荒謬。
- 只根據提供的文字內容出題，不要編造文字稿沒有提到的資訊。

輸出格式：只能輸出一個 JSON 陣列，不要有任何其他文字、說明或 markdown 標記。
陣列中每個元素的格式為：
{"question": "題目文字", "options": ["選項A", "選項B", "選項C", "選項D"], "correctIndex": 0}
correctIndex 是 0-3 的整數，代表 options 陣列中正確答案的索引。
"""


def extract_json_array(text: str):
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Find first '[' and last ']' to be robust against stray text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model output")
    return json.loads(text[start : end + 1])


@app.post("/api/generate-questions")
def generate_questions(req: GenerateRequest):
    title_line = f"影片標題：{req.video_title}\n\n" if req.video_title.strip() else ""
    user_prompt = (
        f"{title_line}"
        f"請根據以下文字稿/重點內容，產生 {req.count} 題四選一問答題：\n\n"
        f"{req.transcript.strip()}"
    )

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"AI 服務呼叫失敗：{exc}") from exc

    raw_text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )

    try:
        questions = extract_json_array(raw_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"AI 回傳格式無法解析：{exc}") from exc

    cleaned = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct_index = q.get("correctIndex", None)
        if (
            question
            and isinstance(options, list)
            and len(options) == 4
            and all(isinstance(o, str) and o.strip() for o in options)
            and isinstance(correct_index, int)
            and 0 <= correct_index <= 3
        ):
            cleaned.append(
                {
                    "question": question,
                    "options": [o.strip() for o in options],
                    "correctIndex": correct_index,
                }
            )

    if not cleaned:
        raise HTTPException(status_code=422, detail="AI 未能產生有效的題目，請嘗試提供更完整的文字稿。")

    return {"questions": cleaned[: req.count]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
