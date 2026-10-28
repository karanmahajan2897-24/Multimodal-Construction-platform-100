"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Single-file Streamlit app. No database: everything lives in st.session_state.
Run locally:   streamlit run app.py
Deploy:        push app.py + requirements.txt to GitHub, then create the app
               on Streamlit Community Cloud. Optionally add GEMINI_API_KEY to
               the app's Secrets; otherwise users paste a key in the sidebar.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
import urllib.parse
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:  # Gemini SDK is optional: without it the app runs in demonstration mode.
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    GENAI_AVAILABLE = False

st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONSTANTS
# =============================================================================
APP_TITLE = "Multimodal Construction Intelligence Platform"
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_OPTIONS = [DEFAULT_MODEL, "gemini-2.5-flash-lite", "gemini-2.5-pro", "Custom…"]
MAX_UPLOAD_MB = 10

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_COLORS = {"Low": "#2E8B57", "Medium": "#B98300", "High": "#D9622B", "Critical": "#C0262D"}
STATUS_ORDER = ["Compliant", "Needs Attention", "Non-Compliant", "Critical Risk"]
STATUS_COLORS = {
    "Compliant": "#2E8B57",
    "Needs Attention": "#B98300",
    "Non-Compliant": "#D9622B",
    "Critical Risk": "#C0262D",
}
PPE_COLORS = {"Detected": "#2E8B57", "Partial": "#B98300", "Missing": "#C0262D", "Not Visible": "#6B7785"}
PPE_ITEMS = [
    ("Helmet", ("helmet", "hard hat", "head")),
    ("High-Vis Vest", ("vest", "hi-vis", "high-vis", "high vis", "visibility")),
    ("Steel-Toe Boots", ("boot", "footwear", "shoe")),
    ("Safety Harness", ("harness", "fall")),
]

DEFECT_TYPES = ["Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling"]
URGENCY_LEVELS = ["Routine Monitoring", "Plan Repair (1-3 months)", "Urgent (within 2 weeks)", "Immediate Action"]
URGENCY_COLORS = ["#2E8B57", "#B98300", "#D9622B", "#C0262D"]

SAFETY_PROMPT = """Analyse the attached construction-site photograph as a senior site-safety auditor.

Standards to apply: OSHA 29 CFR 1926 (construction), India's BOCW Act and Central Rules 1998, IS 7969 (safety in construction), IS 3696 (scaffolds and ladders), IS 3764 (excavation safety), and IS 456:2000 for concrete, formwork and curing practice visible in the frame. Cite a standard only when you are confident it applies; otherwise write "General good practice".

Rules:
- Judge only what is visible. If no worker is visible set workers_detected to 0 and mark every PPE item "Not Visible". Never invent workers or PPE.
- Evaluate exactly four PPE items: Helmet, High-Vis Vest, Steel-Toe Boots, Safety Harness. A harness is only required for work at height (about 1.8 m / 6 ft or more); if nobody is working at height use "Not Visible" and say so in the notes.
- Use status "Partial" when some visible workers comply and others do not, and say how many in the notes.
- Hazards cover housekeeping, edge protection, scaffolds, excavations, electrical, lifting, rebar, formwork and fire risks.
- Scoring guide: 90-100 exemplary, 70-89 minor issues, 40-69 significant non-compliance, below 40 imminent danger.

Return ONLY one JSON object with this exact shape:
{
  "safety_score": <integer 0-100>,
  "overall_status": "Compliant" | "Needs Attention" | "Non-Compliant" | "Critical Risk",
  "workers_detected": <integer>,
  "summary": "<2-3 sentence assessment>",
  "ppe_detection": [
    {"item": "Helmet", "status": "Detected" | "Missing" | "Partial" | "Not Visible", "confidence": <0-1>, "notes": "<short>"},
    {"item": "High-Vis Vest", "status": "...", "confidence": <0-1>, "notes": "<short>"},
    {"item": "Steel-Toe Boots", "status": "...", "confidence": <0-1>, "notes": "<short>"},
    {"item": "Safety Harness", "status": "...", "confidence": <0-1>, "notes": "<short>"}
  ],
  "hazards": [
    {"hazard": "<what is wrong>", "severity": "Low" | "Medium" | "High" | "Critical",
     "location_in_image": "<where>", "standard_reference": "<standard or General good practice>",
     "corrective_remedy": "<specific fix>"}
  ],
  "priority_actions": ["<action 1>", "<action 2>", "<action 3>"]
}"""

CRACK_PROMPT = """Analyse the attached photograph of a concrete or masonry surface as a senior structural-diagnostics engineer working to Indian practice (IS 456:2000, IS 13935 for repair and seismic strengthening, IS 15988 where relevant).

Classify the principal defect into exactly one of: "Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling", or "No Significant Defect".

Guidance:
- Hairline: fine, random or map-pattern cracks, typically below 0.3 mm.
- Settlement: stair-step or diagonal cracks in masonry, or vertical/diagonal cracks widening toward the top or bottom, often with distortion near openings.
- Structural Shear: diagonal (about 45 degrees) cracks near supports or beam-column joints, or X-shaped cracks, in a load-bearing member.
- Thermal Expansion: straight horizontal or vertical cracks at slab edges, parapets, long walls and joints, often cyclic with temperature.
- Spalling: concrete cover breaking away, delamination, exposed or rusting reinforcement.

Rules:
- Estimate crack width in millimetres from visible cues (a coin, ruler, bricks, mortar joints, rebar). If there is no scale reference, give your best estimate and say so in width_basis.
- Describe only what the photo supports. Lower the confidence when the image is blurry or lacks scale.
- Repairs must be specific engineering methods (for example epoxy injection, routing and sealing with polymer-modified sealant, structural re-grouting, jacketing, polymer-modified mortar patch repair with rebar treatment, expansion-joint provision, underpinning).

Return ONLY one JSON object with this exact shape:
{
  "defect_type": "Hairline" | "Settlement" | "Structural Shear" | "Thermal Expansion" | "Spalling" | "No Significant Defect",
  "confidence": <0-1>,
  "estimated_crack_width_mm": <number>,
  "width_basis": "<how the width was estimated>",
  "crack_pattern": "<orientation, length, branching, location on member>",
  "severity": "Low" | "Medium" | "High" | "Critical",
  "urgency": "Routine Monitoring" | "Plan Repair (1-3 months)" | "Urgent (within 2 weeks)" | "Immediate Action",
  "summary": "<2-3 sentence diagnosis>",
  "root_causes": ["<cause 1>", "<cause 2>"],
  "recommended_repairs": [{"method": "<name>", "description": "<how and where>", "priority": "Immediate" | "Short-term" | "Long-term"}],
  "monitoring_plan": ["<step 1>", "<step 2>"],
  "further_investigation": ["<test or survey 1>", "<test or survey 2>"]
}"""

CHAT_SYSTEM = """You are a practical construction-engineering assistant for Indian building sites. You support site engineers, supervisors and students.

Style: concise, structured, plain language. Use short bullet points for procedures. Give numbers (limits, durations, cover, frequencies) when you are confident, and name the governing code (IS 456, IS 800, IS 3696, IS 7969, NBC 2016, OSHA 1926, BOCW rules) only when sure.
Safety first: if a question involves risk to people, lead with the safety control. For structural decisions (removing props, loading slabs, repairing cracks in load-bearing members) say the engineer of record must approve.
If you are unsure or the question lacks needed details, say what you need rather than guessing.
End advice that affects safety or structure with one line: "Confirm against the project specification and your engineer of record."
"""

# =============================================================================
# 2. STYLING
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.hero { background: #0F2A43; color: #FFFFFF; padding: 22px 28px 18px 28px; border-radius: 6px 6px 0 0; }
.hero-title { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif; font-size: 2.5rem; font-weight: 700; line-height: 1.05; }
.hero-sub { color: #C9D6E2; max-width: 74ch; margin-top: 6px; font-size: 1rem; }
.hero-meta { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.hazard-stripe { height: 10px; border-radius: 0 0 6px 6px; margin-bottom: 1.1rem;
  background: repeating-linear-gradient(-45deg, #F2B01E 0 14px, #16202B 14px 28px); }
.section-title { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif; font-size: 1.7rem; font-weight: 700; line-height: 1.1; margin: .2rem 0 .1rem 0; }
.section-sub { opacity: .75; margin-bottom: .8rem; max-width: 90ch; }
.subhead { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif; font-size: 1.3rem; font-weight: 700; margin: 1.1rem 0 .4rem 0; }
.kpi { border: 1px solid rgba(128,128,128,.3); border-top: 4px solid #F2B01E; border-radius: 6px; padding: 12px 14px 10px 14px; background: rgba(128,128,128,.07); height: 100%; }
.kpi-label { font-size: .85rem; opacity: .75; }
.kpi-value { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif; font-size: 2.15rem; font-weight: 700; line-height: 1.1; margin: 2px 0; }
.kpi-unit { font-size: 1rem; opacity: .6; margin-left: 2px; }
.kpi-sub { font-size: .8rem; opacity: .7; }
.pill { display: inline-block; padding: 2px 11px; border-radius: 999px; color: #FFFFFF; font-size: .78rem; font-weight: 600; white-space: nowrap; }
.pill-light { background: rgba(255,255,255,.14); color: #FFFFFF; border: 1px solid rgba(255,255,255,.3); }
.card { border: 1px solid rgba(128,128,128,.3); border-left: 5px solid #888; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; background: rgba(128,128,128,.05); }
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 4px; }
.card-title { font-weight: 600; }
.card-body { font-size: .92rem; }
.card-meta { font-size: .82rem; opacity: .75; margin-top: 2px; }
.bar { height: 10px; background: rgba(128,128,128,.25); border-radius: 5px; overflow: hidden; margin: 10px 0 4px 0; }
.bar-fill { height: 100%; border-radius: 5px; }
.note { border-left: 4px solid #F2B01E; background: rgba(242,176,30,.12); padding: 8px 12px; border-radius: 4px; font-size: .9rem; margin: 6px 0 12px 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 12px 0; }
.chip { padding: 5px 12px; border-radius: 4px; font-size: .85rem; font-weight: 500; border: 1px solid rgba(128,128,128,.35); }
.chip-done { background: #2E8B57; color: #fff; border-color: #2E8B57; }
.chip-now { background: #F2B01E; color: #16202B; border-color: #F2B01E; font-weight: 600; }
.chip-next { background: rgba(128,128,128,.1); }
.side-brand { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif; font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.small-muted { font-size: .82rem; opacity: .7; }
</style>
"""

# =============================================================================
# 3. SMALL HELPERS
# =============================================================================


def esc(value: Any) -> str:
    """HTML-escape model/user text; also neutralise '$' so Streamlit does not treat it as LaTeX."""
    return html.escape(str(value), quote=True).replace("$", "&#36;")


def html_block(markup: str) -> None:
    """Render trusted HTML. Lines are flattened so markdown never sees indented code blocks."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{esc(text)}</span>'


def kpi(label: str, value_html: str, sub: str = "", accent: str = "#F2B01E") -> None:
    html_block(
        f'<div class="kpi" style="border-top-color:{accent}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{value_html}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def section(title: str, sub: str) -> None:
    html_block(f'<div class="section-title">{esc(title)}</div><div class="section-sub">{esc(sub)}</div>')


def subhead(text: str) -> None:
    html_block(f'<div class="subhead">{esc(text)}</div>')


def to_int(v: Any, default: int = 0, lo: int | None = None, hi: int | None = None) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        n = default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def to_float(v: Any, default: float = 0.0, lo: float | None = None, hi: float | None = None) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError):
        n = default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip() or default


def as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if v in (None, ""):
        return []
    return [v]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_inr(value: float) -> str:
    """Compact INR: Crores / Lakhs."""
    v = float(value)
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} Lakh"
    return f"₹{v:,.0f}"


def indian_commas(n: float) -> str:
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def score_color(score: int) -> str:
    if score >= 80:
        return "#2E8B57"
    if score >= 60:
        return "#B98300"
    if score >= 40:
        return "#D9622B"
    return "#C0262D"


def normalize_severity(v: Any, default: str = "Medium") -> str:
    s = as_str(v).lower()
    for sev in SEVERITY_ORDER:
        if sev.lower() in s:
            return sev
    aliases = {"minor": "Low", "info": "Low", "moderate": "Medium", "major": "High", "severe": "High", "extreme": "Critical"}
    for key, sev in aliases.items():
        if key in s:
            return sev
    return default


def parse_json_response(text: str | None) -> dict:
    """Parse JSON from a model reply, tolerating code fences and stray prose."""
    if not text:
        raise ValueError("The model returned an empty response (it may have been blocked by safety filters).")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model reply did not contain valid JSON.")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("The model reply was JSON but not an object.")
    return data


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission_denied" in low or "401" in low or "403" in low:
        return "Gemini rejected the API key. Check that it is correct, active and allowed to use the selected model."
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate" in low:
        return "Gemini rate limit or quota reached. Wait a minute and retry, or switch to a lighter model in the sidebar."
    if "404" in low or "not found" in low:
        return "The selected model was not found for this key. Choose another model in the sidebar."
    if "safety" in low or "blocked" in low:
        return "The request was blocked by Gemini safety filters. Try a different photo."
    if "timeout" in low or "timed out" in low or "connection" in low or "unavailable" in low or "503" in low:
        return "Could not reach Gemini (network or service issue). Please retry."
    return f"The AI request failed: {msg[:240]}"


def prepare_image(uploaded: Any) -> tuple[bytes, Image.Image]:
    """Validate, orient, downscale and re-encode an upload as JPEG."""
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller photo.")
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError("That file could not be read as an image. Upload a JPG or PNG photo.") from exc
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), img


def get_secret_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


# =============================================================================
# 4. AI ENGINE (Gemini live mode)
# =============================================================================
class AIEngine:
    """Thin wrapper over the Google Gen AI SDK. `live` is False in demonstration mode."""

    SYSTEM_VISION = "You are a meticulous construction-engineering analyst. Reply with a single valid JSON object and nothing else."

    def __init__(self, api_key: str, model: str):
        self.api_key = (api_key or "").strip()
        self.model = model or DEFAULT_MODEL
        self._client = None
        if self.api_key and GENAI_AVAILABLE:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def live(self) -> bool:
        return self._client is not None

    @property
    def mode_label(self) -> str:
        return f"Live · {self.model}" if self.live else "Demonstration mode"

    def analyze_image(self, prompt: str, image_bytes: bytes) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=self.SYSTEM_VISION,
            temperature=0.2,
            response_mime_type="application/json",
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=config,
        )
        return parse_json_response(resp.text)

    def generate_text(self, prompt: str, temperature: float = 0.5) -> str:
        config = types.GenerateContentConfig(temperature=temperature)
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        if not resp.text:
            raise ValueError("The model returned an empty response.")
        return resp.text.strip()

    def chat(self, history: list[dict], system_instruction: str) -> str:
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in history[-20:]
        ]
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.4)
        resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        if not resp.text:
            raise ValueError("The model returned an empty response.")
        return resp.text.strip()

    def ping(self) -> None:
        self.generate_text("Reply with the single word: ready", temperature=0.0)


# =============================================================================
# 5. RESULT NORMALISERS (make model output safe to render)
# =============================================================================
def normalize_safety(raw: dict) -> dict:
    score = to_int(raw.get("safety_score"), 50, 0, 100)

    status = as_str(raw.get("overall_status"))
    status = next((s for s in STATUS_ORDER if s.lower() == status.lower()), "")
    if not status:
        status = "Compliant" if score >= 85 else "Needs Attention" if score >= 65 else "Non-Compliant" if score >= 40 else "Critical Risk"

    ppe_raw = [p for p in as_list(raw.get("ppe_detection")) if isinstance(p, dict)]
    ppe: list[dict] = []
    for canonical, keys in PPE_ITEMS:
        match = next((p for p in ppe_raw if any(k in as_str(p.get("item")).lower() for k in keys)), None)
        if match is None:
            ppe.append({"item": canonical, "status": "Not Visible", "confidence": 0.0, "notes": "Not assessed in this image."})
            continue
        s = as_str(match.get("status")).lower()
        if "not visible" in s or "unclear" in s or "n/a" in s or "not assessed" in s or "not required" in s:
            st_name = "Not Visible"
        elif "partial" in s or "some" in s:
            st_name = "Partial"
        elif "miss" in s or "absent" in s or "not worn" in s or "non-compliant" in s or s in ("no", "false"):
            st_name = "Missing"
        elif "detect" in s or "worn" in s or "present" in s or "compliant" in s or s in ("yes", "true"):
            st_name = "Detected"
        else:
            st_name = "Not Visible"
        conf = to_float(match.get("confidence"), 0.0, 0.0, 100.0)
        conf = conf / 100 if conf > 1 else conf
        ppe.append({"item": canonical, "status": st_name, "confidence": round(conf, 2), "notes": as_str(match.get("notes"))})

    hazards: list[dict] = []
    for h in as_list(raw.get("hazards")):
        if not isinstance(h, dict) or not as_str(h.get("hazard")):
            continue
        hazards.append(
            {
                "hazard": as_str(h.get("hazard")),
                "severity": normalize_severity(h.get("severity")),
                "location_in_image": as_str(h.get("location_in_image"), "Not specified"),
                "standard_reference": as_str(h.get("standard_reference"), "General good practice"),
                "corrective_remedy": as_str(h.get("corrective_remedy"), "Assess and correct before work continues."),
            }
        )
    hazards.sort(key=lambda h: -SEVERITY_ORDER.index(h["severity"]))

    if any(h["severity"] == "Critical" for h in hazards) and status in ("Compliant", "Needs Attention"):
        status = "Non-Compliant"

    return {
        "safety_score": score,
        "overall_status": status,
        "workers_detected": to_int(raw.get("workers_detected"), 0, 0),
        "summary": as_str(raw.get("summary"), "No summary provided."),
        "ppe_detection": ppe,
        "hazards": hazards,
        "priority_actions": [as_str(a) for a in as_list(raw.get("priority_actions")) if as_str(a)],
    }


def normalize_crack(raw: dict) -> dict:
    d = as_str(raw.get("defect_type")).lower()
    if "hair" in d:
        defect = "Hairline"
    elif "settle" in d:
        defect = "Settlement"
    elif "shear" in d:
        defect = "Structural Shear"
    elif "thermal" in d or "expansion" in d:
        defect = "Thermal Expansion"
    elif "spall" in d:
        defect = "Spalling"
    elif "no " in d or "none" in d:
        defect = "No Significant Defect"
    else:
        defect = as_str(raw.get("defect_type"), "Unclassified").title()

    severity = normalize_severity(raw.get("severity"))
    u = as_str(raw.get("urgency")).lower()
    if "immediate" in u:
        urgency_idx = 3
    elif "urgent" in u or "2 week" in u or "two week" in u:
        urgency_idx = 2
    elif "plan" in u or "month" in u:
        urgency_idx = 1
    elif "monitor" in u or "routine" in u:
        urgency_idx = 0
    else:
        urgency_idx = SEVERITY_ORDER.index(severity)

    repairs = []
    for r in as_list(raw.get("recommended_repairs")):
        if isinstance(r, dict) and as_str(r.get("method")):
            repairs.append(
                {
                    "method": as_str(r.get("method")),
                    "description": as_str(r.get("description")),
                    "priority": as_str(r.get("priority"), "Short-term"),
                }
            )
        elif isinstance(r, str) and r.strip():
            repairs.append({"method": r.strip(), "description": "", "priority": "Short-term"})

    return {
        "defect_type": defect,
        "confidence": round(to_float(raw.get("confidence"), 0.0, 0.0, 1.0), 2),
        "estimated_crack_width_mm": round(to_float(raw.get("estimated_crack_width_mm"), 0.0, 0.0, 500.0), 2),
        "width_basis": as_str(raw.get("width_basis"), "Visual estimate"),
        "crack_pattern": as_str(raw.get("crack_pattern"), "Not described"),
        "severity": severity,
        "urgency": URGENCY_LEVELS[urgency_idx],
        "summary": as_str(raw.get("summary"), "No summary provided."),
        "root_causes": [as_str(x) for x in as_list(raw.get("root_causes")) if as_str(x)],
        "recommended_repairs": repairs,
        "monitoring_plan": [as_str(x) for x in as_list(raw.get("monitoring_plan")) if as_str(x)],
        "further_investigation": [as_str(x) for x in as_list(raw.get("further_investigation")) if as_str(x)],
    }


def width_band(mm: float) -> tuple[str, str]:
    """Return (label, guidance) for a crack width. 0.3 mm is the usual IS 456 serviceability reference."""
    if mm <= 0:
        return "No measurable crack", "No width-based action required."
    if mm < 0.3:
        return "Within 0.3 mm limit", "Typical serviceability tolerance; seal for durability and monitor."
    if mm < 1.0:
        return "Above 0.3 mm limit", "Exceeds the usual IS 456 limit; seal or inject and find the cause."
    if mm < 5.0:
        return "Significant", "Needs a structural assessment before repair."
    return "Severe", "Treat as potentially structural; restrict use and call a structural engineer."


# =============================================================================
# 6. DEMONSTRATION DATA (used when no API key is supplied or a live call fails)
# =============================================================================
def _ppe(item: str, status: str, conf: float, notes: str) -> dict:
    return {"item": item, "status": status, "confidence": conf, "notes": notes}


def _haz(hazard: str, severity: str, loc: str, std: str, fix: str) -> dict:
    return {"hazard": hazard, "severity": severity, "location_in_image": loc, "standard_reference": std, "corrective_remedy": fix}


def demo_safety(seed: bytes) -> dict:
    idx = int(hashlib.md5(seed or b"demo").hexdigest(), 16) % 3
    scenarios = [
        {
            "safety_score": 62,
            "overall_status": "Non-Compliant",
            "workers_detected": 4,
            "summary": "Most workers wear helmets and vests, but footwear and fall protection are inconsistent. Open slab edges and protruding rebar create avoidable injury risks.",
            "ppe_detection": [
                _ppe("Helmet", "Partial", 0.88, "3 of 4 visible workers wear helmets."),
                _ppe("High-Vis Vest", "Detected", 0.91, "All visible workers wear high-visibility vests."),
                _ppe("Steel-Toe Boots", "Partial", 0.64, "Two workers appear to wear canvas shoes."),
                _ppe("Safety Harness", "Missing", 0.79, "One worker near the slab edge has no harness or anchorage."),
            ],
            "hazards": [
                _haz("Unprotected slab edge with worker nearby", "High", "Upper deck, right side", "OSHA 1926.502 / IS 7969", "Install guardrails (top rail 1 m, mid rail, toe board) before work continues; enforce harness with anchor points."),
                _haz("Exposed rebar ends without caps", "Medium", "Column starters, foreground", "OSHA 1926.701(b)", "Fit mushroom caps or bend over rebar ends within reach of workers."),
                _haz("Scaffold platform without toe boards", "High", "Left scaffold bay", "IS 3696 (Part 1)", "Add toe boards and complete decking; have a competent person tag the scaffold."),
                _haz("Debris and loose timber on walkway", "Low", "Ground level, centre", "General good practice", "Clear and stack materials; assign housekeeping at the end of every shift."),
            ],
            "priority_actions": ["Stop work at the open slab edge until guardrails are in place.", "Issue safety footwear and replace non-compliant shoes today.", "Hold a toolbox talk on working at height."],
        },
        {
            "safety_score": 86,
            "overall_status": "Compliant",
            "workers_detected": 5,
            "summary": "Good PPE discipline across the crew with clear access routes. Only minor housekeeping and cable-management issues were found.",
            "ppe_detection": [
                _ppe("Helmet", "Detected", 0.95, "All five visible workers wear helmets with chin straps."),
                _ppe("High-Vis Vest", "Detected", 0.94, "Vests are clean and reflective strips are visible."),
                _ppe("Steel-Toe Boots", "Detected", 0.82, "Safety boots visible on all workers whose feet are in frame."),
                _ppe("Safety Harness", "Not Visible", 0.5, "No work at height observed, so no harness is required."),
            ],
            "hazards": [
                _haz("Extension cable crossing a walkway", "Medium", "Centre foreground", "General good practice", "Route the cable overhead or use a cable ramp; check insulation for damage."),
                _haz("Cement bags stacked unevenly", "Low", "Left storage area", "IS 7969", "Restack no higher than 10 bags, on pallets, with the stack interlocked."),
            ],
            "priority_actions": ["Re-route the temporary power cable.", "Keep up the current PPE checks and add the stack height rule to the daily briefing."],
        },
        {
            "safety_score": 31,
            "overall_status": "Critical Risk",
            "workers_detected": 3,
            "summary": "Workers are exposed to serious risks: no head protection, no fall arrest at height and an unsupported excavation. Immediate stop-work action is recommended.",
            "ppe_detection": [
                _ppe("Helmet", "Missing", 0.9, "None of the three visible workers wear a helmet."),
                _ppe("High-Vis Vest", "Missing", 0.76, "Workers are in ordinary clothing."),
                _ppe("Steel-Toe Boots", "Missing", 0.6, "Sandals or open footwear appear on one worker."),
                _ppe("Safety Harness", "Missing", 0.85, "A worker is on an upper level with no fall-arrest system."),
            ],
            "hazards": [
                _haz("Deep trench with no shoring, sloping or barricade", "Critical", "Foreground excavation", "OSHA 1926.652 / IS 3764", "Stop entry. Provide shoring or benching, a safe ladder every 7.5 m, and edge barricades with spoil kept back."),
                _haz("Work at height with no fall protection", "Critical", "Upper level, rear", "OSHA 1926.501 / IS 7969", "Remove workers from the edge; install guardrails or a certified lifeline system before restart."),
                _haz("Temporary wiring with exposed conductors", "High", "Near the water pool", "IS 732 / General good practice", "Isolate the supply, replace with armoured cable and an RCCB, and keep it clear of water."),
                _haz("Poor housekeeping and tripping hazards", "Medium", "Throughout the site", "General good practice", "Clear walkways and assign a housekeeping crew."),
            ],
            "priority_actions": ["Suspend work at the trench and upper level immediately.", "Issue helmets, vests and boots before anyone re-enters the site.", "Have a competent person inspect the site before restart."],
        },
    ]
    return scenarios[idx]


def demo_crack(seed: bytes) -> dict:
    idx = int(hashlib.md5(seed or b"demo").hexdigest(), 16) % 5
    base = [
        {
            "defect_type": "Hairline", "confidence": 0.81, "estimated_crack_width_mm": 0.15,
            "width_basis": "Estimated against surface texture; no scale reference in the frame.",
            "crack_pattern": "Fine random map pattern across the plaster/concrete skin, no preferred direction.",
            "severity": "Low", "urgency": "Routine Monitoring",
            "summary": "Fine crazing cracks consistent with early shrinkage. No sign of structural distress or moisture staining.",
            "root_causes": ["Plastic or drying shrinkage from rapid moisture loss", "Inadequate or late curing", "High cement or water content in the surface layer"],
            "recommended_repairs": [
                {"method": "Surface sealing", "description": "Brush in a polymer-modified cementitious slurry or acrylic elastomeric coating to close the cracks.", "priority": "Long-term"},
                {"method": "Curing correction", "description": "Apply wet curing or a curing compound on any adjacent fresh work.", "priority": "Short-term"},
            ],
            "monitoring_plan": ["Mark crack ends with a date and re-inspect after 4 weeks.", "Use a crack-width card at the same spot each time."],
            "further_investigation": ["Check the curing records for this pour.", "Tap the surface for hollow sounds that indicate delamination."],
        },
        {
            "defect_type": "Settlement", "confidence": 0.77, "estimated_crack_width_mm": 3.5,
            "width_basis": "Compared with the thickness of the mortar joints in the masonry.",
            "crack_pattern": "Diagonal stair-step crack running through the mortar joints from the opening corner toward the ground, wider at the top.",
            "severity": "High", "urgency": "Urgent (within 2 weeks)",
            "summary": "The stair-step pattern and a width that increases toward one end point to differential foundation settlement. The cause should be established before cosmetic repair.",
            "root_causes": ["Differential settlement of the footing on weak or poorly compacted soil", "Soil moisture change or leaking drainage near the foundation", "Nearby excavation or loss of lateral support"],
            "recommended_repairs": [
                {"method": "Crack gauge survey", "description": "Fix tell-tales or digital gauges across the crack to learn whether movement is still active.", "priority": "Immediate"},
                {"method": "Underpinning or ground improvement", "description": "If movement continues, design underpinning (mass concrete or micropiles) or compaction grouting.", "priority": "Short-term"},
                {"method": "Stitching and re-pointing", "description": "After stabilisation, stitch the wall with helical bars and re-point with non-shrink mortar.", "priority": "Long-term"},
            ],
            "monitoring_plan": ["Record readings weekly for 8 weeks and after every heavy rain.", "Photograph the crack from the same angle each time."],
            "further_investigation": ["Trial pit or borehole to check the founding soil.", "Levelling survey of plinth and floor.", "Inspect drains, pipes and water sources near the foundation."],
        },
        {
            "defect_type": "Structural Shear", "confidence": 0.72, "estimated_crack_width_mm": 2.5,
            "width_basis": "Compared with visible reinforcement spacing; no scale reference.",
            "crack_pattern": "Inclined crack at about 45 degrees running from the support face toward mid-depth of the beam.",
            "severity": "Critical", "urgency": "Immediate Action",
            "summary": "A diagonal crack near the support of a load-carrying member is a classic shear-distress signature. This can lead to a brittle failure with little warning.",
            "root_causes": ["Insufficient shear reinforcement (stirrup spacing or size)", "Overloading beyond the design load", "Poor concrete strength or honeycombing at the support", "Reinforcement congestion or displaced stirrups"],
            "recommended_repairs": [
                {"method": "Temporary propping", "description": "Prop the member immediately and restrict loading or access underneath.", "priority": "Immediate"},
                {"method": "Epoxy injection", "description": "Pressure-inject low-viscosity epoxy to restore monolithic action once the structural engineer confirms the cause.", "priority": "Short-term"},
                {"method": "Strengthening by jacketing", "description": "Add shear capacity with a reinforced concrete jacket or externally bonded FRP/steel plates, as designed by the structural engineer.", "priority": "Short-term"},
            ],
            "monitoring_plan": ["Keep the area cleared and propped until a structural engineer signs off.", "Install crack monitors and check daily."],
            "further_investigation": ["Rebound hammer and UPV tests, plus core cutting for concrete strength.", "Cover meter or GPR scan to confirm stirrup layout.", "Structural re-analysis against actual loads."],
        },
        {
            "defect_type": "Thermal Expansion", "confidence": 0.74, "estimated_crack_width_mm": 1.2,
            "width_basis": "Estimated from the edge sharpness and the adjacent brick course.",
            "crack_pattern": "Straight, near-horizontal crack at the slab-wall junction, uniform in width along its length.",
            "severity": "Medium", "urgency": "Plan Repair (1-3 months)",
            "summary": "The straight, uniform crack at a restrained junction fits cyclic thermal movement. It is mainly a durability and water-ingress issue, not an immediate structural risk.",
            "root_causes": ["Daily and seasonal temperature swings on an exposed roof slab", "No expansion joint or sliding layer between slab and wall", "Insufficient roof insulation or reflective finish"],
            "recommended_repairs": [
                {"method": "Rout and seal", "description": "Cut a V-groove, clean it and fill with a flexible polyurethane or polysulphide sealant over a backer rod.", "priority": "Short-term"},
                {"method": "Expansion joint provision", "description": "Introduce movement joints at suitable spacing on long walls and roof slabs.", "priority": "Long-term"},
                {"method": "Thermal protection", "description": "Apply a reflective or insulating roof treatment to reduce the temperature range.", "priority": "Long-term"},
            ],
            "monitoring_plan": ["Measure the width in the hottest and coolest weeks to confirm cyclic behaviour.", "Check for leakage marks after the first rain."],
            "further_investigation": ["Review the joint layout in the drawings.", "Check the waterproofing layer near the crack."],
        },
        {
            "defect_type": "Spalling", "confidence": 0.84, "estimated_crack_width_mm": 2.0,
            "width_basis": "Estimated from the delaminated edge against the exposed bar diameter.",
            "crack_pattern": "Concrete cover has broken away along the soffit with rust-stained, exposed reinforcement and flaking edges.",
            "severity": "High", "urgency": "Urgent (within 2 weeks)",
            "summary": "Loss of cover with exposed, corroding rebar indicates active reinforcement corrosion. Expansion of rust will keep pushing the cover off until the cause is treated.",
            "root_causes": ["Inadequate concrete cover to reinforcement", "Carbonation or chloride-induced corrosion", "Water ingress from a leaking slab or poor drainage", "Poor-quality or highly permeable concrete"],
            "recommended_repairs": [
                {"method": "Patch repair", "description": "Break out loose concrete, clean the rebar to Sa 2 (or replace if section loss exceeds 15 percent), apply an anti-corrosive primer, and repair with polymer-modified repair mortar.", "priority": "Short-term"},
                {"method": "Protective coating", "description": "Apply an anti-carbonation or waterproof coating to the repaired area and surrounding concrete.", "priority": "Long-term"},
                {"method": "Source of moisture", "description": "Fix leaks and drainage to prevent the process restarting.", "priority": "Immediate"},
            ],
            "monitoring_plan": ["Sound the surrounding area for further delamination.", "Inspect after the monsoon and again after 6 months."],
            "further_investigation": ["Carbonation depth test (phenolphthalein).", "Half-cell potential and chloride content tests.", "Cover meter survey of adjacent members."],
        },
    ]
    return base[idx]


DEMO_KB: list[tuple[tuple[str, ...], str]] = [
    (("curing", "cure"),
     "**Concrete curing (IS 456, Cl. 13.5)**\n- Keep exposed surfaces continuously damp for at least **7 days** with ordinary Portland cement and at least **10 days** with mineral admixtures or blended cement.\n- Use **at least 10 days** in hot, dry weather; extending to 14 days is recommended for blended cements.\n- Methods: ponding, wet hessian, or a curing compound on vertical faces.\n\nConfirm against the project specification and your engineer of record."),
    (("formwork", "shuttering", "stripping", "de-shutter", "deshutter", "props"),
     "**Typical formwork striking times (IS 456, Cl. 11.3, OPC, around 15 °C or above)**\n- Vertical faces of columns, walls and beams: **16-24 hours**\n- Slab soffit (props refixed immediately): **3 days**\n- Beam soffit: **7 days**\n- Props to slabs: **7 days** up to 4.5 m span, **14 days** beyond\n- Props to beams: **14 days** up to 6 m span, **21 days** beyond\n\nLonger times apply in cold weather or with slower cements. Confirm against cube results, the project specification and your engineer of record."),
    (("cube", "sample", "sampling", "strength test"),
     "**Cube sampling frequency (IS 456, Table 11)**\n- 1-5 m³: 1 sample\n- 6-15 m³: 2 samples\n- 16-30 m³: 3 samples\n- 31-50 m³: 4 samples\n- Above 50 m³: 4 samples plus 1 for each extra 50 m³ or part thereof\n\nEach sample = 3 cubes; test at 28 days (and 7 days as an early indicator). Confirm against the project specification and your engineer of record."),
    (("cover", "clear cover"),
     "**Nominal cover to reinforcement (IS 456, Table 16, general)**\n- Mild: 20 mm · Moderate: 30 mm · Severe: 45 mm · Very severe: 50 mm · Extreme: 75 mm\n- Columns need at least 40 mm (or bar diameter, whichever is greater) under normal conditions.\n- Use cover blocks of the same grade as the concrete and fix them at proper spacing.\n\nConfirm against the project specification and your engineer of record."),
    (("slump", "workability", "mix", "water cement", "w/c"),
     "**Workability and mix points**\n- Typical slump: 25-75 mm for lightly reinforced sections, 50-100 mm for heavily reinforced, and 100-150 mm for pumped concrete (follow the approved mix design).\n- Never add water on site to improve workability; use a plasticiser as per the mix design.\n- Correct for aggregate moisture, especially after rain.\n\nConfirm against the project specification and your engineer of record."),
    (("crane", "lifting", "hoist"),
     "**Crane and lifting safety**\n- Follow the load chart and the manufacturer's wind limit (commonly around 30-36 km/h sustained for lifting).\n- Hold a lift plan, trained signaller, and an exclusion zone under the load.\n- Do not lift when visibility is poor or lightning is nearby.\n- Inspect slings, shackles and hooks before each shift.\n\nConfirm against the project specification and your engineer of record."),
    (("scaffold", "height", "fall", "harness", "ladder"),
     "**Work at height checklist**\n- Scaffold: base plates, bracing, ties, full decking, guardrails and toe boards, with a tag after inspection by a competent person (IS 3696).\n- Fall protection from roughly 1.8 m (6 ft): guardrails first, harness with a certified anchor point where rails are impractical.\n- Ladders: secured at the top, 1 m beyond the landing, 4:1 angle.\n\nConfirm against the project specification and your engineer of record."),
    (("trench", "excavation", "shoring"),
     "**Excavation safety (IS 3764, OSHA 1926 Subpart P)**\n- Shore, slope or bench any trench deeper than about 1.5 m; a competent person inspects daily and after rain.\n- Keep spoil and equipment back from the edge; provide a ladder within 7.5 m of workers.\n- Locate underground services before digging and keep water out.\n\nConfirm against the project specification and your engineer of record."),
    (("ppe", "helmet", "vest", "boots", "protective"),
     "**Minimum PPE on an active site**\n- Safety helmet with chin strap, high-visibility vest, safety shoes with steel toe and midsole.\n- Add gloves, eye protection, hearing protection and respirators for the task (cutting, grinding, mixing).\n- Harness for work at height, and PPE inspected and replaced when damaged.\n\nConfirm against the project specification and your engineer of record."),
    (("crack", "cracks"),
     "**Crack triage on site**\n- Under 0.3 mm and random: usually shrinkage, seal and monitor.\n- Diagonal near supports or widening: stop loading and call the structural engineer.\n- Stair-step in masonry: look for settlement; fix tell-tales first.\n- Straight cracks at joints and slab edges: thermal movement; rout and seal.\n\nUse the Crack Diagnostics tab to analyse a photo. Confirm against the project specification and your engineer of record."),
    (("rain", "monsoon", "wet"),
     "**Working in rain**\n- Stop concreting unless the pour is covered; protect fresh concrete with tarpaulin.\n- De-energise temporary electrical panels in wet areas and use tested RCCBs.\n- Inspect trenches and slopes before re-entry; pump out water.\n- Store cement on raised pallets under cover; correct mixes for wet aggregate.\n\nConfirm against the project specification and your engineer of record."),
    (("heat", "hot", "summer", "temperature"),
     "**Working in extreme heat**\n- Schedule heavy work for early morning or evening; set water-rest-shade cycles and a buddy system.\n- Concrete: pour early, use chilled water or ice and retarders, and keep placement temperature within the specification (IS 7861 Part 1).\n- Begin curing immediately and continue for at least 10 days.\n\nConfirm against the project specification and your engineer of record."),
]


def demo_answer(question: str) -> str:
    q = question.lower()
    for keys, answer in DEMO_KB:
        if any(k in q for k in keys):
            return answer
    return (
        "I'm running in **demonstration mode**, so I can only answer from a built-in reference set. "
        "Try asking about concrete curing, formwork striking times, cube sampling, cover, slump, cranes, scaffolds, "
        "excavations, PPE, cracks, rain or heat.\n\nAdd a Gemini API key in the sidebar for open-ended answers."
    )


# =============================================================================
# 7. PLANNING & VALUATION MODEL (pure Python)
# =============================================================================
STRUCTURE_TYPES: dict[str, dict[str, Any]] = {
    "RCC framed building": {"cost_factor": 1.00, "market_factor": 1.00, "steel_kg_sqft": 4.0, "cement_bags_sqft": 0.45,
                            "desc": "reinforced concrete framed residential building"},
    "Load-bearing masonry": {"cost_factor": 0.85, "market_factor": 0.90, "steel_kg_sqft": 2.5, "cement_bags_sqft": 0.40,
                             "desc": "load-bearing brick masonry house"},
    "Steel / PEB structure": {"cost_factor": 1.15, "market_factor": 0.95, "steel_kg_sqft": 11.0, "cement_bags_sqft": 0.18,
                              "desc": "pre-engineered steel structure building"},
    "Independent villa": {"cost_factor": 1.35, "market_factor": 1.25, "steel_kg_sqft": 4.5, "cement_bags_sqft": 0.50,
                          "desc": "luxury independent villa"},
    "Commercial complex": {"cost_factor": 1.45, "market_factor": 1.40, "steel_kg_sqft": 5.0, "cement_bags_sqft": 0.50,
                           "desc": "commercial complex with glass facade"},
}
LOCATION_TIERS: dict[str, dict[str, Any]] = {
    "Tier 1 (Mumbai, Delhi NCR, Bengaluru, Hyderabad, Chennai)": {"build_psf": 2600, "market_psf": 9500, "land_share": 0.40,
                                                                  "setting": "dense metropolitan Indian city skyline in the background"},
    "Tier 2 (Pune, Jaipur, Lucknow, Coimbatore, Nagpur)": {"build_psf": 2100, "market_psf": 5800, "land_share": 0.30,
                                                           "setting": "growing Indian city neighbourhood with a tree-lined road"},
    "Tier 3 (smaller cities and towns)": {"build_psf": 1750, "market_psf": 3200, "land_share": 0.22,
                                          "setting": "small Indian town surroundings with greenery"},
}
SOFT_COST_RATE = 0.12  # design, approvals, supervision and contingency on top of hard cost
# (phase, start %, end %, cost weight, steel weight, cement weight)
PHASES = [
    ("Foundation and earthwork", 0, 15, 0.12, 0.20, 0.15),
    ("Plinth and basement", 15, 25, 0.08, 0.10, 0.10),
    ("RCC superstructure", 25, 60, 0.35, 0.65, 0.40),
    ("Masonry and rough MEP", 60, 80, 0.20, 0.05, 0.20),
    ("Finishing and fixtures", 80, 97, 0.22, 0.00, 0.14),
    ("Handover and snagging", 97, 100, 0.03, 0.00, 0.01),
]


def phase_fraction(completion: float, start: float, end: float) -> float:
    return max(0.0, min(1.0, (completion - start) / (end - start)))


def estimate_project(structure: str, sqft: float, tier: str, completion: float,
                     build_psf: float | None = None, market_psf: float | None = None,
                     land_share: float | None = None) -> dict:
    s, t = STRUCTURE_TYPES[structure], LOCATION_TIERS[tier]
    build_psf = build_psf if build_psf else t["build_psf"] * s["cost_factor"]
    market_psf = market_psf if market_psf else t["market_psf"] * s["market_factor"]
    land_share = t["land_share"] if land_share is None else land_share

    hard_cost = sqft * build_psf
    budget = hard_cost * (1 + SOFT_COST_RATE)
    cost_frac = steel_frac = cement_frac = 0.0
    for _, a, b, wc, ws, wm in PHASES:
        f = phase_fraction(completion, a, b)
        cost_frac += wc * f
        steel_frac += ws * f
        cement_frac += wm * f

    capex = budget * cost_frac
    market_full = sqft * market_psf
    valuation = market_full * (land_share + (1 - land_share) * cost_frac)
    steel_total_t = sqft * s["steel_kg_sqft"] / 1000
    cement_total_bags = sqft * s["cement_bags_sqft"]
    current_phase = next((p[0] for p in PHASES if completion < p[2]), PHASES[-1][0]) if completion < 100 else "Completed"
    return {
        "build_psf": build_psf, "market_psf": market_psf, "land_share": land_share,
        "hard_cost": hard_cost, "budget": budget, "cost_frac": cost_frac, "capex": capex,
        "remaining": budget - capex, "market_full": market_full, "valuation": valuation,
        "steel_total_t": steel_total_t, "steel_used_t": steel_total_t * steel_frac,
        "cement_total_bags": cement_total_bags, "cement_used_bags": cement_total_bags * cement_frac,
        "current_phase": current_phase,
    }


def build_render_prompt(structure: str, floors: int, style: str, tier: str, completion: int) -> str:
    if completion < 15:
        stage = "early stage with excavation, foundation pits and earthmovers"
    elif completion < 25:
        stage = "plinth beams and column starter bars under construction"
    elif completion < 60:
        stage = "RCC frame rising with shuttering, scaffolding and a tower crane"
    elif completion < 80:
        stage = "brick masonry and facade work in progress behind steel scaffolding"
    elif completion < 97:
        stage = "finishing stage with glazing installed and exterior paint nearly done"
    else:
        stage = "fully completed, landscaped and ready for handover"
    return (
        f"Architectural 3D visualisation of a {floors}-storey {STRUCTURE_TYPES[structure]['desc']}, {style} style, "
        f"{stage}, {LOCATION_TIERS[tier]['setting']}, golden hour lighting, wide-angle exterior view, "
        "photorealistic, highly detailed"
    )


def pollinations_url(prompt: str, seed: int) -> str:
    return f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt[:900])}?width=1280&height=720&nologo=true&seed={seed}"


# =============================================================================
# 8. WEATHER ADVISORY DATA
# =============================================================================
ACTIVITIES = ["Crane and lifting", "Work at height", "Concreting", "Excavation and trenching",
              "Electrical works", "Welding and hot work", "Painting and finishing"]
GO, CAUTION, STOP = "Go", "Caution", "Stop"
GO_COLORS = {GO: "#2E8B57", CAUTION: "#B98300", STOP: "#C0262D"}

WEATHER: dict[str, dict[str, Any]] = {
    "Normal": {
        "level": "Low",
        "headline": "Normal conditions: standard site controls apply.",
        "alerts": [
            ("Daily routine", "Run the toolbox talk, check permits to work and confirm the day's lift plans."),
            ("Housekeeping", "Keep walkways, stairs and scaffold platforms clear; stack materials on pallets."),
            ("Watch the forecast", "Re-check conditions at midday; weather can change quickly on exposed sites."),
            ("Inspections", "Inspect scaffolds, excavations and lifting gear at the start of each shift."),
        ],
        "curing": ["Keep concrete surfaces damp for at least 7 days with OPC and 10 days with blended cement (IS 456).",
                   "Start curing as soon as the surface can take it without damage."],
        "matrix": {},
    },
    "High Wind": {
        "level": "High",
        "headline": "High wind: suspend lifting and work at height until it drops below the limits.",
        "alerts": [
            ("Crane and hoist", "Stop lifting when sustained wind reaches the limit on the crane's load chart (commonly around 30-36 km/h). Lower the load, slew the jib to weathervane, and secure the crane."),
            ("Work at height", "Stop work on scaffolds, roof edges and facades. Secure sheeting, tarpaulins, formwork panels and loose boards."),
            ("Materials", "Tie down rebar bundles, sheets and pallets, and keep people out of areas below tall stacks."),
            ("Temporary structures", "Check scaffold ties, hoardings, signboards and site offices for anchorage and damage after strong gusts."),
        ],
        "curing": ["Wind dries the surface quickly: fit windbreaks and cover slabs with wet hessian or plastic.",
                   "Apply a curing compound straight after finishing and look for plastic shrinkage cracks in the first hours."],
        "matrix": {
            "Crane and lifting": (STOP, "Above the load-chart wind limit; park and secure the crane."),
            "Work at height": (STOP, "Falling and wind-borne object risk."),
            "Concreting": (CAUTION, "Windbreaks and fast curing needed."),
            "Excavation and trenching": (CAUTION, "Dust and flying debris; wet down spoil."),
            "Electrical works": (CAUTION, "Check overhead lines and temporary supports."),
            "Welding and hot work": (STOP, "Sparks can spread fire in the wind."),
            "Painting and finishing": (CAUTION, "Overspray and fast drying; use screens."),
        },
    },
    "Heavy Rain": {
        "level": "High",
        "headline": "Heavy rain: electrical, excavation and concrete controls must be in place before work resumes.",
        "alerts": [
            ("Electrical hazard", "De-energise temporary panels in wet areas, check RCCB trip tests, lift cables out of water and use only dry, trained electricians."),
            ("Excavations", "Pump out water and have a competent person re-inspect trench walls, shoring and slopes before anyone re-enters."),
            ("Concreting", "Do not place concrete in rain unless the pour is covered. Drain water from the pour, protect fresh surfaces, and correct mixes for wet aggregate."),
            ("Slips and access", "Stop work on wet scaffold decks and ladders; put down anti-slip walkways and mark flooded areas."),
            ("Materials", "Keep cement on raised pallets under cover and protect rebar, sand and brick stocks from washout and waterlogging."),
        ],
        "curing": ["Cover fresh concrete with tarpaulin; rain falling on a setting surface can wash out the cement paste.",
                   "Inspect surfaces for washout afterwards. No extra curing water is needed while it stays wet, but do not strike formwork early."],
        "matrix": {
            "Crane and lifting": (CAUTION, "Slippery ground, poor visibility; check bearing and visibility."),
            "Work at height": (STOP, "Wet, slippery surfaces and lightning risk."),
            "Concreting": (STOP, "Only continue with full cover and drainage."),
            "Excavation and trenching": (STOP, "Collapse risk; re-inspect before entry."),
            "Electrical works": (STOP, "Shock risk outdoors; isolate wet circuits."),
            "Welding and hot work": (STOP, "Electric shock and unstable footing."),
            "Painting and finishing": (STOP, "Moisture will ruin finishes and adhesion."),
        },
    },
    "Extreme Heat": {
        "level": "High",
        "headline": "Extreme heat: protect workers from heat illness and change the concrete plan.",
        "alerts": [
            ("Worker health", "Set water-rest-shade cycles, move heavy work to early morning or evening, use a buddy system and watch for heat exhaustion or stroke signs. Provide ORS and shaded rest areas."),
            ("Hot-weather concreting", "Pour early or at night, use chilled water or ice and retarding admixtures, shade aggregates, and keep placing temperature within the specification (IS 7861 Part 1)."),
            ("Curing", "Start curing immediately and keep going for at least 10 days in hot, dry weather (IS 456). Use wet hessian, ponding or curing compound."),
            ("Equipment and materials", "Watch for overheating plant, hot steel and formwork surfaces. Store fuel, gas and adhesives out of direct sun."),
        ],
        "curing": ["Place curing materials before the pour starts so there is no gap after finishing.",
                   "Cure at least 10 days, keep the surface continuously wet, and consider shading to cut evaporation."],
        "matrix": {
            "Crane and lifting": (GO, "Give operators and signallers regular heat breaks."),
            "Work at height": (CAUTION, "Limit exposure time; heat illness raises fall risk."),
            "Concreting": (CAUTION, "Hot-weather concreting controls apply."),
            "Excavation and trenching": (CAUTION, "Limit hours, provide shade and water."),
            "Electrical works": (GO, "Standard controls; avoid hot panels in direct sun."),
            "Welding and hot work": (CAUTION, "Fire risk and added heat load; run a fire watch."),
            "Painting and finishing": (CAUTION, "Fast drying; work in shade or at cooler hours."),
        },
    },
}
WEATHER_LEVEL_COLORS = {"Low": "#2E8B57", "Medium": "#B98300", "High": "#D9622B", "Critical": "#C0262D"}
QUICK_PROMPTS = [
    "How long should I cure concrete in hot weather?",
    "When can I strike slab formwork?",
    "What PPE is mandatory on site?",
    "How often should I take concrete cube samples?",
]


# =============================================================================
# 9. SIDEBAR
# =============================================================================
ARCH_TEXT = """Streamlit UI (app.py)
 ├─ Sidebar: API key · model · mode
 ├─ Tab 1 Safety audit ─┐
 ├─ Tab 2 Crack diag.  ─┼─► AIEngine ─► Gemini (vision, JSON mode)
 ├─ Tab 4 Assistant    ─┘      └─► Demo engine (offline fallback)
 └─ Tab 3 Planning: Python estimator
          + Pollinations image URL
Session state only · no database"""


def render_sidebar() -> AIEngine:
    with st.sidebar:
        html_block('<div class="side-brand">Construction Intelligence</div><div class="small-muted">Skill Wallet Capstone Project 1</div>')
        st.divider()
        st.markdown("**Gemini connection**")
        secret_key = get_secret_key()
        typed = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="Paste your key to enable live AI",
            help="Create a free key in Google AI Studio. The key is kept only in this browser session and is never stored.",
            label_visibility="collapsed",
        ).strip()
        api_key = typed or secret_key

        model_choice = st.selectbox("Model", MODEL_OPTIONS, index=0)
        model = DEFAULT_MODEL
        if model_choice == "Custom…":
            model = st.text_input("Custom model ID", value=DEFAULT_MODEL).strip() or DEFAULT_MODEL
        else:
            model = model_choice

        engine = AIEngine(api_key, model)
        if engine.live:
            src = "Streamlit secrets" if (secret_key and not typed) else "sidebar"
            st.success(f"Live mode on ({src}).")
            if st.button("Verify key"):
                try:
                    with st.spinner("Checking key…"):
                        engine.ping()
                    st.success("The key works.")
                except Exception as exc:
                    st.error(friendly_error(exc))
        else:
            if api_key and not GENAI_AVAILABLE:
                st.warning("The google-genai package is not installed, so the app is in demonstration mode.")
            else:
                st.info("Demonstration mode: add an API key above for real image analysis and chat.")

        st.divider()
        st.markdown("**Modules**")
        st.markdown(
            "- **Site safety and PPE audit** (Epic 2)\n"
            "- **Crack diagnostics** (Epic 2)\n"
            "- **Planning and valuation** (Epic 3)\n"
            "- **Daily site assistant** (Epic 5)"
        )
        st.caption("Switch modules with the tabs at the top of the page.")

        with st.expander("Architecture overview"):
            st.code(ARCH_TEXT, language="text")
            st.markdown(
                "- Photos are resized and sent to Gemini only in live mode.\n"
                "- Replies are requested as JSON, then validated before display.\n"
                "- If a live call fails, the app shows the error and a demonstration result."
            )
        with st.expander("Privacy and limits"):
            st.markdown(
                "- Nothing is saved to a database.\n"
                "- Results are screening aids, not certifications. A qualified engineer must confirm structural and safety decisions."
            )
        if st.button("Reset session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    return engine


def render_hero(engine: AIEngine) -> None:
    mode_color = "#2E8B57" if engine.live else "#B98300"
    html_block(
        f'<div class="hero"><div class="hero-title">{esc(APP_TITLE)}</div>'
        '<div class="hero-sub">Photo-based safety audits and crack diagnosis, project budgeting and valuation, '
        'and a weather-aware site assistant, in one workspace.</div>'
        f'<div class="hero-meta">{pill(engine.mode_label, mode_color)}'
        '<span class="pill pill-light">OSHA · IS 456 · BOCW</span></div></div>'
        '<div class="hazard-stripe"></div>'
    )


def init_state() -> None:
    defaults = {"safety_result": None, "crack_result": None, "chat": [], "project_ctx": "", "render": None, "pending_prompt": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# =============================================================================
# 10. TAB 1 - SAFETY & PPE AUDIT
# =============================================================================
def run_analysis(engine: AIEngine, prompt: str, image_bytes: bytes | None, demo_fn: Callable[[bytes], dict],
                 normalizer: Callable[[dict], dict], spinner: str) -> tuple[dict, str]:
    """Run live analysis when possible; fall back to the demo result with a clear message."""
    seed = image_bytes or b"demo"
    if engine.live and image_bytes:
        try:
            with st.spinner(spinner):
                raw = engine.analyze_image(prompt, image_bytes)
            return normalizer(raw), "live"
        except Exception as exc:
            st.error(friendly_error(exc))
            st.warning("Showing a demonstration result instead so you can still explore the layout.")
    return normalizer(demo_fn(seed)), "demo"


def tab_safety(engine: AIEngine) -> None:
    section("Site safety and PPE audit",
            "Upload a site photo to check helmets, vests, boots and harnesses and to list hazards with fixes (OSHA, IS 456 and related Indian codes).")
    left, right = st.columns([1, 1])
    with left:
        upload = st.file_uploader("Construction site photo", type=["jpg", "jpeg", "png"], key="safety_upload")
        notes = st.text_area("Site notes (optional)", placeholder="Example: slab casting on level 3, work started at 7 am", key="safety_notes", height=90)
    image_bytes, preview = None, None
    if upload is not None:
        try:
            image_bytes, preview = prepare_image(upload)
        except ValueError as exc:
            st.error(str(exc))
    with right:
        if preview is not None:
            st.image(preview, caption="Uploaded photo")
        else:
            st.caption("The photo preview appears here.")

    needs_image = engine.live
    disabled = needs_image and image_bytes is None
    if disabled:
        st.caption("Upload a photo to run the audit.")
    elif not engine.live:
        st.caption("Demonstration mode: this returns a sample report. Upload a photo to vary the sample.")
    if st.button("Run safety audit", type="primary", disabled=disabled, key="run_safety"):
        prompt = SAFETY_PROMPT + (f"\n\nSite notes from the supervisor: {notes.strip()}" if notes.strip() else "")
        result, mode = run_analysis(engine, prompt, image_bytes, demo_safety, normalize_safety, "Auditing site safety…")
        st.session_state.safety_result = {"result": result, "mode": mode, "model": engine.model, "at": now_iso()}

    saved = st.session_state.safety_result
    if saved:
        render_safety_report(saved)


def render_safety_report(saved: dict) -> None:
    res = saved["result"]
    score = res["safety_score"]
    color = score_color(score)
    if saved["mode"] == "demo":
        html_block('<div class="note">This is a <b>demonstration result</b>, not an analysis of your photo.</div>')

    subhead("Audit summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Safety score", f'{score}<span class="kpi-unit">/ 100</span>', "Higher is safer", color)
    with c2:
        kpi("Overall status", pill(res["overall_status"], STATUS_COLORS[res["overall_status"]]), "Based on PPE and hazards", STATUS_COLORS[res["overall_status"]])
    with c3:
        kpi("Workers detected", str(res["workers_detected"]), "Visible in the photo", "#0F2A43")
    crit = sum(1 for h in res["hazards"] if h["severity"] in ("Critical", "High"))
    with c4:
        kpi("Hazards found", str(len(res["hazards"])), f"{crit} high or critical", SEVERITY_COLORS["High"] if crit else SEVERITY_COLORS["Low"])
    html_block(f'<div class="bar"><div class="bar-fill" style="width:{score}%;background:{color}"></div></div>')
    st.write(res["summary"])

    subhead("PPE checklist")
    cols = st.columns(4)
    for col, item in zip(cols, res["ppe_detection"]):
        with col:
            conf = f'Confidence {int(item["confidence"] * 100)}%' if item["confidence"] else ""
            html_block(
                f'<div class="card" style="border-left-color:{PPE_COLORS[item["status"]]}">'
                f'<div class="card-head"><span class="card-title">{esc(item["item"])}</span>{pill(item["status"], PPE_COLORS[item["status"]])}</div>'
                f'<div class="card-body">{esc(item["notes"]) or "&nbsp;"}</div>'
                f'<div class="card-meta">{esc(conf)}</div></div>'
            )

    subhead("Identified hazards")
    if not res["hazards"]:
        st.success("No hazards were identified in this photo.")
    for h in res["hazards"]:
        c = SEVERITY_COLORS[h["severity"]]
        html_block(
            f'<div class="card" style="border-left-color:{c}">'
            f'<div class="card-head"><span class="card-title">{esc(h["hazard"])}</span>{pill(h["severity"], c)}</div>'
            f'<div class="card-meta">Where: {esc(h["location_in_image"])} · Standard: {esc(h["standard_reference"])}</div>'
            f'<div class="card-body"><b>Corrective action:</b> {esc(h["corrective_remedy"])}</div></div>'
        )

    if res["priority_actions"]:
        subhead("Priority actions")
        for a in res["priority_actions"]:
            st.markdown(f"- {a}")

    report = {"report_type": "site_safety_ppe_audit", "generated_at": saved["at"], "mode": saved["mode"], "model": saved["model"], **res}
    st.download_button("Download JSON report", json.dumps(report, indent=2, ensure_ascii=False),
                       file_name=f"safety_audit_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json", key="dl_safety")


# =============================================================================
# 11. TAB 2 - CRACK DIAGNOSTICS
# =============================================================================
def tab_crack(engine: AIEngine) -> None:
    section("Crack diagnostics and structural integrity",
            "Upload a photo of a concrete or masonry surface to classify the defect, estimate crack width, and get causes and repair methods.")
    left, right = st.columns([1, 1])
    with left:
        upload = st.file_uploader("Surface photo", type=["jpg", "jpeg", "png"], key="crack_upload")
        member = st.selectbox("Structural member", ["Not sure", "Beam", "Column", "Slab", "Load-bearing wall", "Partition wall", "Foundation or plinth", "Parapet or roof"], key="crack_member")
        ctx = st.text_input("Scale or context (optional)", placeholder="Example: ₹5 coin next to the crack; building is 6 years old", key="crack_ctx")
    image_bytes, preview = None, None
    if upload is not None:
        try:
            image_bytes, preview = prepare_image(upload)
        except ValueError as exc:
            st.error(str(exc))
    with right:
        if preview is not None:
            st.image(preview, caption="Uploaded photo")
        else:
            st.caption("The photo preview appears here.")

    disabled = engine.live and image_bytes is None
    if disabled:
        st.caption("Upload a photo to run the diagnosis.")
    elif not engine.live:
        st.caption("Demonstration mode: this returns a sample diagnosis. Upload a photo to vary the sample.")
    if st.button("Diagnose defect", type="primary", disabled=disabled, key="run_crack"):
        extra = f"\n\nStructural member: {member}." + (f" Context: {ctx.strip()}" if ctx.strip() else "")
        result, mode = run_analysis(engine, CRACK_PROMPT + extra, image_bytes, demo_crack, normalize_crack, "Diagnosing the defect…")
        st.session_state.crack_result = {"result": result, "mode": mode, "model": engine.model, "at": now_iso(), "member": member}

    saved = st.session_state.crack_result
    if saved:
        render_crack_report(saved)


def render_crack_report(saved: dict) -> None:
    r = saved["result"]
    sev_c = SEVERITY_COLORS[r["severity"]]
    urg_i = URGENCY_LEVELS.index(r["urgency"])
    if saved["mode"] == "demo":
        html_block('<div class="note">This is a <b>demonstration result</b>, not an analysis of your photo.</div>')

    subhead("Diagnosis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Defect type", esc(r["defect_type"]), f'Confidence {int(r["confidence"] * 100)}%', sev_c)
    with c2:
        kpi("Estimated width", f'{r["estimated_crack_width_mm"]:g}<span class="kpi-unit">mm</span>', width_band(r["estimated_crack_width_mm"])[0], sev_c)
    with c3:
        kpi("Severity", pill(r["severity"], sev_c), "Structural significance", sev_c)
    with c4:
        kpi("Urgency", pill(r["urgency"], URGENCY_COLORS[urg_i]), "Time to act", URGENCY_COLORS[urg_i])

    st.write(r["summary"])
    band_label, band_note = width_band(r["estimated_crack_width_mm"])
    html_block(
        f'<div class="note"><b>Width check:</b> {esc(band_label)}. {esc(band_note)} '
        f'<span class="small-muted">Basis: {esc(r["width_basis"])}. Photo-based widths are rough; confirm with a crack-width gauge.</span></div>'
    )
    st.markdown(f"**Crack pattern:** {r['crack_pattern']}")

    col_a, col_b = st.columns(2)
    with col_a:
        subhead("Likely root causes")
        for cause in r["root_causes"] or ["Not determined from the photo."]:
            st.markdown(f"- {cause}")
    with col_b:
        subhead("Monitoring plan")
        for step in r["monitoring_plan"] or ["No monitoring steps provided."]:
            st.markdown(f"- {step}")

    subhead("Recommended engineering repairs")
    prio_colors = {"immediate": "#C0262D", "short-term": "#D9622B", "long-term": "#2E8B57"}
    for rep in r["recommended_repairs"] or []:
        pc = prio_colors.get(rep["priority"].lower(), "#6B7785")
        html_block(
            f'<div class="card" style="border-left-color:{pc}">'
            f'<div class="card-head"><span class="card-title">{esc(rep["method"])}</span>{pill(rep["priority"], pc)}</div>'
            f'<div class="card-body">{esc(rep["description"])}</div></div>'
        )
    if r["further_investigation"]:
        subhead("Further investigation")
        for item in r["further_investigation"]:
            st.markdown(f"- {item}")

    st.caption("AI-assisted visual screening only. A licensed structural engineer must inspect and approve any repair to a load-bearing member.")
    report = {"report_type": "crack_diagnostics", "generated_at": saved["at"], "mode": saved["mode"], "model": saved["model"],
              "structural_member": saved.get("member", ""), **r}
    st.download_button("Download JSON report", json.dumps(report, indent=2, ensure_ascii=False),
                       file_name=f"crack_diagnosis_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json", key="dl_crack")


# =============================================================================
# 12. TAB 3 - PLANNING & VALUATION
# =============================================================================
def tab_planning(engine: AIEngine) -> None:
    section("Planning and real estate valuation",
            "Estimate budget, spend to date, current asset value and material needs for an Indian project. Figures are planning-level estimates, not a valuation certificate.")
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        structure = st.selectbox("Structure type", list(STRUCTURE_TYPES), key="plan_structure")
    with c2:
        sqft = st.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000, value=2400, step=100, key="plan_sqft")
    with c3:
        tier = st.selectbox("Location tier", list(LOCATION_TIERS), index=1, key="plan_tier")
    completion = st.slider("Project completion (%)", 0, 100, 45, key="plan_completion")

    defaults = estimate_project(structure, sqft, tier, completion)
    with st.expander("Adjust cost and market assumptions"):
        a1, a2, a3 = st.columns(3)
        k = f"{structure}|{tier}"
        with a1:
            build_psf = st.number_input("Construction cost (₹ per sq ft)", min_value=500, max_value=20000,
                                        value=int(round(defaults["build_psf"])), step=50, key=f"bp_{k}")
        with a2:
            market_psf = st.number_input("Market rate (₹ per sq ft)", min_value=500, max_value=100000,
                                         value=int(round(defaults["market_psf"])), step=100, key=f"mp_{k}")
        with a3:
            land_pct = st.slider("Land share of market value (%)", 0, 80, int(round(defaults["land_share"] * 100)), key=f"ls_{k}")
        st.caption(f"Soft costs (design, approvals, supervision, contingency) add {int(SOFT_COST_RATE * 100)}% to construction cost.")
    est = estimate_project(structure, sqft, tier, completion, build_psf, market_psf, land_pct / 100)

    st.session_state.project_ctx = (
        f"{structure}, {int(sqft)} sq ft, {tier.split(' (')[0]}, {completion}% complete (phase: {est['current_phase']}), "
        f"total budget {fmt_inr(est['budget'])}"
    )

    subhead("Financial snapshot")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi("Total project budget", esc(fmt_inr(est["budget"])), f"₹{indian_commas(est['budget'])}", "#0F2A43")
    with m2:
        kpi("Capital expenditure incurred", esc(fmt_inr(est["capex"])), f"{est['cost_frac'] * 100:.0f}% of budget spent", "#F2B01E")
    with m3:
        kpi("Current asset valuation", esc(fmt_inr(est["valuation"])), f"₹{indian_commas(est['valuation'])}", "#2E8B57")
    with m4:
        kpi("Value at completion", esc(fmt_inr(est["market_full"])), f"₹{indian_commas(est['market_psf'])} per sq ft", "#2E8B57")

    subhead("Materials and remaining spend")
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        kpi("Steel requirement", f'{est["steel_total_t"]:.1f}<span class="kpi-unit">tonnes</span>', f'{est["steel_used_t"]:.1f} t used · {est["steel_total_t"] - est["steel_used_t"]:.1f} t left', "#6B7785")
    with n2:
        kpi("Cement requirement", f'{indian_commas(est["cement_total_bags"])}<span class="kpi-unit">bags</span>',
            f'{indian_commas(est["cement_used_bags"])} used · {indian_commas(est["cement_total_bags"] - est["cement_used_bags"])} left (50 kg bags)', "#6B7785")
    with n3:
        kpi("Remaining spend", esc(fmt_inr(est["remaining"])), "To finish the project", "#D9622B")
    with n4:
        kpi("Current phase", esc(est["current_phase"]), f"{completion}% physical progress", "#0F2A43")

    subhead("Milestone progress")
    st.progress(completion / 100, text=f"{completion}% complete")
    chips = []
    rows = []
    for name, a, b, wc, _, _ in PHASES:
        if completion >= b:
            cls, status = "chip-done", "Done"
        elif completion >= a:
            cls, status = "chip-now", "In progress"
        else:
            cls, status = "chip-next", "Upcoming"
        chips.append(f'<span class="chip {cls}">{esc(name)}</span>')
        rows.append({"Phase": name, "Progress range": f"{a}-{b}%", "Budget share": f"{wc * 100:.0f}%",
                     "Phase budget": fmt_inr(est["budget"] * wc), "Status": status})
    html_block(f'<div class="chip-row">{"".join(chips)}</div>')
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("How these numbers are calculated"):
        st.markdown(
            "- **Budget** = area × construction rate × (1 + 12% soft costs). Rates depend on structure type and city tier.\n"
            "- **Capex incurred** weights each phase by its typical share of cost, so spend does not rise in a straight line with progress.\n"
            "- **Current valuation** = market value at completion × (land share + building share × cost completed).\n"
            "- **Steel and cement** use per-sq-ft norms by structure type and are spread across phases by typical consumption.\n"
            "- Rates are indicative averages; use the assumptions panel to match your local quotes."
        )

    render_visualisation(engine, structure, tier, completion)


def render_visualisation(engine: AIEngine, structure: str, tier: str, completion: int) -> None:
    subhead("Conceptual 3D rendering")
    st.caption("Builds a prompt from your project inputs and generates an image through Pollinations AI (free, no key needed). The image is illustrative only.")
    r1, r2 = st.columns(2)
    with r1:
        floors = st.slider("Number of floors", 1, 20, 2, key="render_floors")
    with r2:
        style = st.selectbox("Architectural style", ["Contemporary Indian", "Modern minimalist", "Neo-classical", "Sustainable green building", "Industrial"], key="render_style")
    auto_prompt = build_render_prompt(structure, floors, style, tier, completion)
    key_sig = hashlib.md5(auto_prompt.encode()).hexdigest()[:8]
    prompt = st.text_area("Image prompt (editable)", value=auto_prompt, height=100, key=f"render_prompt_{key_sig}")
    refine = False
    if engine.live:
        refine = st.checkbox("Polish the prompt with Gemini first", key="render_refine")

    if st.button("Generate rendering", type="primary", key="gen_render"):
        final = prompt.strip() or auto_prompt
        if refine:
            try:
                with st.spinner("Polishing the prompt…"):
                    final = engine.generate_text(
                        "Rewrite this as one vivid text-to-image prompt for an architectural visualisation, under 80 words, "
                        f"keeping every fact. Return only the prompt.\n\n{final}")
            except Exception as exc:
                st.warning(friendly_error(exc) + " Using your original prompt.")
        st.session_state.render = {"prompt": final, "url": pollinations_url(final, int(time.time()) % 100000)}

    data = st.session_state.render
    if data:
        st.image(data["url"], caption="Conceptual rendering via Pollinations AI")
        st.markdown(f"[Open the image in a new tab]({data['url']})")
        with st.expander("Prompt used"):
            st.write(data["prompt"])


# =============================================================================
# 13. TAB 4 - WEATHER-AWARE ASSISTANT
# =============================================================================
def tab_assistant(engine: AIEngine) -> None:
    section("Daily site assistant and weather-aware advisory",
            "Pick today's weather to see alerts, a go/stop view of planned activities, and a curing protocol. Then ask the assistant anything about the work.")
    weather_key = st.radio("Weather condition", list(WEATHER), horizontal=True, key="weather_choice")
    w = WEATHER[weather_key]
    lvl_c = WEATHER_LEVEL_COLORS[w["level"]]
    risk_pill = pill("Risk level: " + w["level"], lvl_c)
    html_block(
        f'<div class="card" style="border-left-color:{lvl_c}"><div class="card-head">'
        f'<span class="card-title">{esc(w["headline"])}</span>{risk_pill}</div></div>'
    )

    subhead("Safety alerts")
    cols = st.columns(2)
    for i, (cat, msg) in enumerate(w["alerts"]):
        with cols[i % 2]:
            html_block(
                f'<div class="card" style="border-left-color:{lvl_c}"><div class="card-title">{esc(cat)}</div>'
                f'<div class="card-body">{esc(msg)}</div></div>'
            )

    left, right = st.columns([1.1, 1])
    with left:
        subhead("Planned activities: go or stop")
        planned = st.multiselect("Today's activities", ACTIVITIES, default=ACTIVITIES[:3], key="planned_activities")
        for act in planned:
            status, note = w["matrix"].get(act, (GO, "Standard controls apply."))
            html_block(
                f'<div class="card" style="border-left-color:{GO_COLORS[status]}"><div class="card-head">'
                f'<span class="card-title">{esc(act)}</span>{pill(status, GO_COLORS[status])}</div>'
                f'<div class="card-meta">{esc(note)}</div></div>'
            )
    with right:
        subhead("Concrete curing protocol")
        for line in w["curing"]:
            st.markdown(f"- {line}")

    st.divider()
    subhead("Engineering assistant")
    render_chat(engine, weather_key)


def render_chat(engine: AIEngine, weather_key: str) -> None:
    if not engine.live:
        st.caption("Demonstration mode: answers come from a built-in reference set. Add a Gemini key for open-ended answers.")
    st.caption("Quick questions")
    qcols = st.columns(len(QUICK_PROMPTS))
    for col, q in zip(qcols, QUICK_PROMPTS):
        with col:
            st.button(q, key=f"qp_{q}", on_click=lambda q=q: st.session_state.__setitem__("pending_prompt", q))

    box = st.container(height=440, border=True)
    typed = st.chat_input("Ask about concrete, safety, scheduling or materials…", key="assistant_input")
    prompt = typed or st.session_state.pop("pending_prompt", None)

    with box:
        if not st.session_state.chat and not prompt:
            st.markdown("Ask a question to get started. Today's weather setting is included automatically.")
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = answer_question(engine, weather_key)
                st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})

    if st.session_state.chat and st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat = []
        st.rerun()


def answer_question(engine: AIEngine, weather_key: str) -> str:
    history = st.session_state.chat
    if engine.live:
        context = f"Today's site weather: {weather_key}."
        if st.session_state.project_ctx:
            context += f" Project in the Planning tab: {st.session_state.project_ctx}."
        try:
            return engine.chat(history, CHAT_SYSTEM + "\n" + context)
        except Exception as exc:
            return f"**{friendly_error(exc)}**\n\n" + demo_answer(history[-1]["content"])
    return demo_answer(history[-1]["content"])


# =============================================================================
# 14. MAIN
# =============================================================================
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    engine = render_sidebar()
    render_hero(engine)

    t1, t2, t3, t4 = st.tabs(["Site safety and PPE", "Crack diagnostics", "Planning and valuation", "Daily site assistant"])
    with t1:
        tab_safety(engine)
    with t2:
        tab_crack(engine)
    with t3:
        tab_planning(engine)
    with t4:
        tab_assistant(engine)

    st.divider()
    st.caption("Outputs are decision-support aids. They do not replace site inspection, structural design or a registered valuer. "
               "Verify every safety-critical or structural decision with a qualified engineer.")


main()
