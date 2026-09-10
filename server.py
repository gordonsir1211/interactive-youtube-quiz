#!/usr/bin/env python3
"""server.py — AI question generation backend for the YouTube quiz page. Runs on port 8000."""
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Anthropic()


class GenerateRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=20000)
    count: int = Field(3, ge=1, le=10)
    video_title: str = ""
    video_duration_seconds: int = Field(0, ge=0, le=36000)


SYSTEM_PROMPT = """你是一位教學設計助理，負責依據老師提供的 YouTube 影片文字稿或內容重點，
設計會在「影片播放過程中」於特定時間點暫停、讓學生作答的四選一多項選擇題，
用來確認學生是否理解剛看過的內容。

規則：
- 每一題必須有且只有一個正確答案，四個選項中只有一個正確。
- 題目與選項一律使用繁體中文（除非文字稿本身是其他語言，此時可保留專有名詞原文）。
- 題目應涵蓋文字稿中不同的重點，依照內容在影片中出現的先後順序排列，避免重複問同一件事。
- 選項長度接近、避免明顯過長或過短的正確答案洩漏答案。
- 干擾選項（錯誤選項）要合理、相關，不能明顯荒謬。
- 只根據提供的文字內容出題，不要編造文字稿沒有提到的資訊。

時間點（timestampSeconds）規則 — 非常重要：
- 如果文字稿內含時間戳記（例如 [00:15]、00:15、1:23:45、(1:05) 等格式），請根據該題內容對應到文字稿中「提到這個重點的時間點」，
  將其換算為總秒數填入 timestampSeconds。應該設定在該重點「剛講解完」的時間點，讓學生看完該段落後立刻作答。
- 如果文字稿完全沒有時間戳記，但有提供影片總長度（video_duration_seconds），
  請依照該重點在文字稿中出現的相對位置（例如文字稿的第幾段/第幾百分比處），
  按比例推算一個合理的 timestampSeconds（0 到 video_duration_seconds 之間），讓多題的時間點依序分散在整部影片中，不要全部擠在同一時間。
- 如果既沒有時間戳記也沒有影片總長度，請將每一題平均分配在 0 到 600 秒（10 分鐘）之間，依題目順序遞增。
- 第一題的 timestampSeconds 不應該是 0（除非文字稿明確指出重點就在影片最開頭），應反映該重點實際出現的時間。
- 多題的 timestampSeconds 必須嚴格遞增（後面的題目時間點一定大於前面的題目）。

輸出格式：只能輸出一個 JSON 陣列，不要有任何其他文字、說明或 markdown 標記。
陣列中每個元素的格式為：
{"question": "題目文字", "options": ["選項A", "選項B", "選項C", "選項D"], "correctIndex": 0, "timestampSeconds": 45}
correctIndex 是 0-3 的整數，代表 options 陣列中正確答案的索引。
timestampSeconds 是非負整數，代表影片中應暫停出題的秒數。
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
    duration_line = (
        f"影片總長度：{req.video_duration_seconds} 秒\n\n"
        if req.video_duration_seconds > 0
        else ""
    )
    user_prompt = (
        f"{title_line}"
        f"{duration_line}"
        f"請根據以下文字稿/重點內容，產生 {req.count} 題四選一問答題，並為每一題標註應在影片中暫停出題的 timestampSeconds：\n\n"
        f"{req.transcript.strip()}"
    )

    try:
        message = client.messages.create(
            model="claude_sonnet_4_6",
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
    last_ts = -1
    for q in questions:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct_index = q.get("correctIndex", None)
        raw_ts = q.get("timestampSeconds", None)
        try:
            timestamp_seconds = int(raw_ts)
        except (TypeError, ValueError):
            timestamp_seconds = None
        if (
            question
            and isinstance(options, list)
            and len(options) == 4
            and all(isinstance(o, str) and o.strip() for o in options)
            and isinstance(correct_index, int)
            and 0 <= correct_index <= 3
            and timestamp_seconds is not None
            and timestamp_seconds >= 0
        ):
            # Enforce strictly increasing timestamps in case the model slips up.
            if timestamp_seconds <= last_ts:
                timestamp_seconds = last_ts + 15
            last_ts = timestamp_seconds
            cleaned.append(
                {
                    "question": question,
                    "options": [o.strip() for o in options],
                    "correctIndex": correct_index,
                    "timestampSeconds": timestamp_seconds,
                }
            )

    if not cleaned:
        raise HTTPException(status_code=422, detail="AI 未能產生有效的題目，請嘗試提供更完整的文字稿。")

    return {"questions": cleaned[: req.count]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
