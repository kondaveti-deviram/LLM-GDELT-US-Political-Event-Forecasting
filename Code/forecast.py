# forecast.py

import json
import re
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def forecast_summary(summary_text: str, count: int, domain_name: str):
    """
    Returns (probability: float, explanation: str)
    """
    system = (
    "You are an experienced intelligence analyst. Last week in the U.S. we saw "
    f"{count} events: {summary_text}\n\n"
    "Your tasks:\n"
    "  1) Forecast the probability (float between 0.0 and 1.0) that at least one "
    f"{domain_name} event occurs next week.\n"
    "  2) Identify which specific events from last week's list most strongly "
    "influence your forecast. Write in 1-2 lines about them.\n\n"
    "Respond *only* with a JSON object with two keys:\n"
    "  • \"prob\": the probability number,\n"
    "  • \"why\": Mention the key events provided of the last week and 1-2 lines justifying why they influenced your forecast. It should be within 300 characters.\n"
    "Example:\n"
    '{\n'
    '  "prob": 0.72,\n'
    '  "why": ...\n'
    "}"
)


    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": summary_text}
        ],
        temperature=0.0
    )
    text = resp.choices[0].message.content.strip()

    # Expect something like: {"prob": 0.42, "why": "Because X and Y..."}
    m = re.search(r'\{.*\}', text, re.S)
    data = {}
    if m:
        try:
            data = json.loads(m.group())
        except:
            pass

    prob = float(data.get("prob", 0))
    why  = data.get("why", "").strip()
    # fallback parsing if JSON didn't work:
    if not why:
        why = text

    return prob, why
