"""Social Listening AI Analysis API — Python backend for LLM-powered insights."""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import base64
import io
from datetime import datetime, timezone
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agentic_chat import run_agentic_chat  # noqa: E402

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def _deepseek_text_sync(prompt: str, max_tokens: int = 8192) -> str:
    """Synchronous DeepSeek call — run via asyncio.to_thread to avoid blocking the event loop."""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=120.0,
    )
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


async def _deepseek_text(prompt: str, max_tokens: int = 8192) -> str:
    return await asyncio.to_thread(_deepseek_text_sync, prompt, max_tokens)

# ---------------------------------------------------------------------------
# IndoBERT sentiment model — lazy-loaded on first request
# ---------------------------------------------------------------------------

_sentiment_pipeline = None
_sentiment_lock = asyncio.Lock()
_sentiment_warmup_task: asyncio.Task | None = None
# Use the git-cloned path when running in Docker, HuggingFace hub otherwise
_INDOBERT_MODEL = (
    "/indonesia-bert-sentiment-classification"
    if os.path.isdir("/indonesia-bert-sentiment-classification")
    else "mdhugol/indonesia-bert-sentiment-classification"
)
_LABEL_MAP = {"LABEL_0": "Positive", "LABEL_1": "Neutral", "LABEL_2": "Negative"}
_SENTIMENT_MAX_BATCH = int(os.getenv("SENTIMENT_MAX_BATCH", "100"))
_SENTIMENT_MAX_CHARS = int(os.getenv("SENTIMENT_MAX_CHARS", "2048"))
_SENTIMENT_INFERENCE_CHARS = int(os.getenv("SENTIMENT_INFERENCE_CHARS", "512"))
_SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "16"))
_TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))


def _load_indobert_sync():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    torch.set_num_threads(_TORCH_NUM_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(_INDOBERT_MODEL)
    # PyTorch 2.6 changed weights_only default True, breaking legacy .bin checkpoints.
    # Patch torch.load directly — works regardless of transformers version.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = AutoModelForSequenceClassification.from_pretrained(_INDOBERT_MODEL)
        model.eval()
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    finally:
        torch.load = _orig_load


async def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    async with _sentiment_lock:
        if _sentiment_pipeline is not None:
            return _sentiment_pipeline
        logger.info("Loading IndoBERT sentiment model…")
        _sentiment_pipeline = await asyncio.to_thread(_load_indobert_sync)
        logger.info("IndoBERT model ready.")
    return _sentiment_pipeline


def _log_sentiment_warmup_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("IndoBERT warmup failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sentiment_warmup_task
    _sentiment_warmup_task = asyncio.create_task(_get_sentiment_pipeline())
    _sentiment_warmup_task.add_done_callback(_log_sentiment_warmup_result)
    yield


app = FastAPI(title="Social Listening AI Analysis API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentimentCount(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class AccountSnapshot(BaseModel):
    followers: int = 0
    posts_scraped: int = 0
    avg_likes: int = 0
    total_engagement: int = 0
    sentiment: SentimentCount = SentimentCount()


class AnalysisRequest(BaseModel):
    kalventis: AccountSnapshot
    gsk: AccountSnapshot
    top_topics: list[str] = []
    news_count: int = 0
    period: str = ""
    top_words: list[str] = []
    follower_ratio: float = 1.0
    post_ratio: float = 1.0


class AnalysisResponse(BaseModel):
    executive_summary: str
    kalventis_insights: str
    gsk_insights: str
    recommendations: list[str]
    risk_indicators: list[str]
    opportunities: list[str]
    sentiment_chart_b64: str | None = None
    engagement_chart_b64: str | None = None
    wordcloud_chart_b64: str | None = None




def _extract_json(raw: str) -> dict:
    """Extract the first balanced JSON object from an LLM response.

    Handles markdown fences, surrounding text, trailing commas, and
    invisible characters (BOM, zero-width spaces).  Uses a character-
    counting brace tracker instead of a greedy regex so that the
    response can contain multiple ``{…}`` blocks without confusion.
    """
    if not raw:
        raise json.JSONDecodeError("Empty LLM response", "", 0)

    cleaned = raw.strip().lstrip("\ufeff")

    # Fast path: already clean JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.replace("```", "")

    # Find the first balanced top-level ``{ … }`` block.
    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("No opening brace in LLM response", cleaned, 0)

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                return json.loads(candidate)

    raise json.JSONDecodeError(
        f"Unbalanced braces in LLM response (length={len(cleaned)})",
        cleaned[:200],
        0,
    )




sns.set_theme(style="whitegrid", palette="muted")


def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib Figure as a base64 data-URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_sentiment_chart(
    sentiment_a: "SentimentCount",
    sentiment_b: "SentimentCount",
    label_a: str,
    label_b: str,
) -> str:
    """Grouped bar chart: positive / neutral / negative for two brands."""
    categories = ["Positive", "Neutral", "Negative"]
    brand_a_vals = [sentiment_a.positive, sentiment_a.neutral, sentiment_a.negative]
    brand_b_vals = [sentiment_b.positive, sentiment_b.neutral, sentiment_b.negative]

    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_a = ax.bar([i - width / 2 for i in x], brand_a_vals, width, label=label_a, color="#3b82f6")
    bars_b = ax.bar([i + width / 2 for i in x], brand_b_vals, width, label=label_b, color="#f59e0b")

    ax.set_ylabel("Post Count")
    ax.set_title("Sentiment Distribution by Brand")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _render_engagement_chart(
    acc_a: "AccountSnapshot",
    acc_b: "AccountSnapshot",
    label_a: str,
    label_b: str,
) -> str:
    """Grouped bar chart comparing engagement metrics for two brands."""
    metrics = ["Followers", "Total\nEngagement", "Avg Likes"]
    vals_a = [acc_a.followers, acc_a.total_engagement, acc_a.avg_likes]
    vals_b = [acc_b.followers, acc_b.total_engagement, acc_b.avg_likes]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_a = ax.bar([i - width / 2 for i in x], vals_a, width, label=label_a, color="#3b82f6")
    bars_b = ax.bar([i + width / 2 for i in x], vals_b, width, label=label_b, color="#f59e0b")

    ax.set_ylabel("Count")
    ax.set_title("Engagement Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def _fmt(v: int) -> str:
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}K"
        return str(v)

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                _fmt(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                _fmt(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64



def _render_wordcloud_chart(words: list[str]) -> str | None:
    """Generate a word cloud PNG from a list of words.

    Uses positional weighting: earlier words get higher frequency so they
    appear larger.  Returns ``None`` when the word list is empty.
    """
    if not words:
        return None

    # Build frequency dict with positional weighting (first = heaviest).
    n = len(words)
    freqs: dict[str, float] = {}
    for i, word in enumerate(words):
        w = word.strip().lower()
        if not w:
            continue
        weight = max(1.0, (n - i) * (100.0 / n))
        freqs[w] = freqs.get(w, 0) + weight

    if not freqs:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        max_words=60,
        background_color="white",
        colormap="viridis",
        collocations=False,
    ).generate_from_frequencies(freqs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64



def _render_sentiment_chart_modern(
    sentiment_a: "SentimentCount",
    sentiment_b: "SentimentCount",
    label_a: str,
    label_b: str,
) -> str:
    """Modern horizontal grouped bar chart shown to the frontend."""
    categories = ["Positive", "Neutral", "Negative"]
    vals_a = [sentiment_a.positive, sentiment_a.neutral, sentiment_a.negative]
    vals_b = [sentiment_b.positive, sentiment_b.neutral, sentiment_b.negative]
    total_a = max(1, sum(vals_a))
    total_b = max(1, sum(vals_b))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fc")

    y = list(range(len(categories)))
    height = 0.32

    bars_a = ax.barh([i + height / 2 for i in y], vals_a, height,
                     label=label_a, color="#2557d6", alpha=0.90)
    bars_b = ax.barh([i - height / 2 for i in y], vals_b, height,
                     label=label_b, color="#12a594", alpha=0.90)

    max_val = max(max(vals_a, default=1), max(vals_b, default=1), 1)

    for bar, val, tot in zip(bars_a, vals_a, [total_a] * 3):
        ax.text(bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val}  ({val / tot * 100:.0f}%)",
                va="center", fontsize=9, color="#374151")
    for bar, val, tot in zip(bars_b, vals_b, [total_b] * 3):
        ax.text(bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val}  ({val / tot * 100:.0f}%)",
                va="center", fontsize=9, color="#374151")

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Post Count", fontsize=10, color="#6b7280")
    ax.set_title("Sentiment Distribution by Brand", fontsize=14, fontweight="bold",
                 color="#111827", pad=16)
    ax.set_xlim(0, max_val * 1.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#d1d5db")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _render_engagement_chart_modern(
    acc_a: "AccountSnapshot",
    acc_b: "AccountSnapshot",
    label_a: str,
    label_b: str,
) -> str:
    """Modern three-panel engagement comparison shown to the frontend."""

    def _fmt(v: int) -> str:
        if v >= 1_000_000: return f"{v / 1_000_000:.1f}M"
        if v >= 1_000: return f"{v / 1_000:.1f}K"
        return str(v)

    panels = [
        ("Followers", acc_a.followers, acc_b.followers),
        ("Total Engagement", acc_a.total_engagement, acc_b.total_engagement),
        ("Avg Likes / Post", acc_a.avg_likes, acc_b.avg_likes),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.patch.set_facecolor("#ffffff")

    for ax, (title, val_a, val_b) in zip(axes, panels):
        ax.set_facecolor("#f8f9fc")
        total = max(1, val_a + val_b)
        bars = ax.bar(
            [label_a, label_b], [val_a, val_b],
            color=["#2557d6", "#12a594"], alpha=0.88, width=0.45,
        )
        for bar, val in zip(bars, [val_a, val_b]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.025,
                _fmt(val), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#111827",
            )
        ax.set_title(title, fontsize=11, fontweight="bold", color="#374151", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Engagement Metrics — {label_a} vs {label_b}",
        fontsize=13, fontweight="bold", color="#111827", y=1.02,
    )
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analysis")
async def generate_analysis(request: AnalysisRequest) -> AnalysisResponse:
    if not DEEPSEEK_API_KEY:
        return AnalysisResponse(
            executive_summary="DeepSeek API key not configured. Add DEEPSEEK_API_KEY to .env to enable AI analysis.",
            kalventis_insights="Configure DEEPSEEK_API_KEY to enable Kalventis insights.",
            gsk_insights="Configure DEEPSEEK_API_KEY to enable GSK competitive insights.",
            recommendations=["Add DEEPSEEK_API_KEY to D:\\fastapi_all\\python-social-listening\\.env"],
            risk_indicators=[],
            opportunities=[],
        )

    try:
        kv = request.kalventis
        gsk = request.gsk
        kv_total = kv.sentiment.positive + kv.sentiment.neutral + kv.sentiment.negative
        gsk_total = gsk.sentiment.positive + gsk.sentiment.neutral + gsk.sentiment.negative
        kv_pos_rate = f"{(kv.sentiment.positive / max(1, kv_total) * 100):.1f}%"
        gsk_pos_rate = f"{(gsk.sentiment.positive / max(1, gsk_total) * 100):.1f}%"

        prompt = f"""You are a senior social media analyst for Kalventis, an Indonesian vaccine awareness brand (@kenapaharusvaksin).

Analyze the following social listening data and provide strategic insights. Be specific, data-driven, and actionable.

=== MONITORING DATA ({request.period}) ===

KALVENTIS (@kenapaharusvaksin) — Owned Brand:
- Followers: {kv.followers:,}
- Posts scraped: {kv.posts_scraped}
- Avg likes/post: {kv.avg_likes}
- Total engagement: {kv.total_engagement:,}
- Sentiment: {kv.sentiment.positive} positive / {kv.sentiment.neutral} neutral / {kv.sentiment.negative} negative → {kv_pos_rate} positive rate

GSK (@ayokitavaksin) — Competitor:
- Followers: {gsk.followers:,}
- Posts scraped: {gsk.posts_scraped}
- Avg likes/post: {gsk.avg_likes}
- Total engagement: {gsk.total_engagement:,}
- Sentiment: {gsk.sentiment.positive} positive / {gsk.sentiment.neutral} neutral / {gsk.sentiment.negative} negative → {gsk_pos_rate} positive rate

COMPETITIVE RATIOS:
- Follower ratio: Kalventis is {request.follower_ratio:.1f}x larger than GSK
- Post volume ratio: Kalventis is {request.post_ratio:.1f}x more active than GSK

MARKET CONTEXT:
- Active vaccine topics: {', '.join(request.top_topics) if request.top_topics else 'None tracked'}
- News articles monitored: {request.news_count}
- Most mentioned terms: {', '.join(request.top_words[:12]) if request.top_words else 'Not available'}

=== INSTRUCTIONS ===
Respond ONLY in valid JSON with exactly these keys (no markdown, no code blocks):
{{
  "executive_summary": "2-3 sentence overall summary of competitive landscape and Kalventis position",
  "kalventis_insights": "2-3 sentences on Kalventis performance, content effectiveness, and audience engagement",
  "gsk_insights": "2-3 sentences on GSK competitive posture and what Kalventis team should know",
  "recommendations": ["specific action 1", "specific action 2", "specific action 3", "specific action 4"],
  "risk_indicators": ["specific risk 1", "specific risk 2"],
  "opportunities": ["growth opportunity 1", "growth opportunity 2", "growth opportunity 3"]
}}

Focus on vaccine awareness, public health education in Indonesia, and practical content strategy advice."""

        raw = await _deepseek_text(prompt)
        data = _extract_json(raw)


        sentiment_b64 = _render_sentiment_chart(kv.sentiment, gsk.sentiment, "Kalventis", "GSK")
        engagement_b64 = _render_engagement_chart(kv, gsk, "Kalventis", "GSK")

        wordcloud_b64 = _render_wordcloud_chart(request.top_words)
        return AnalysisResponse(
            executive_summary=data.get("executive_summary", ""),
            kalventis_insights=data.get("kalventis_insights", ""),
            gsk_insights=data.get("gsk_insights", ""),
            recommendations=data.get("recommendations", []),
            risk_indicators=data.get("risk_indicators", []),
            opportunities=data.get("opportunities", []),
            sentiment_chart_b64=sentiment_b64,
            engagement_chart_b64=engagement_b64,
            wordcloud_chart_b64=wordcloud_b64,
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from DeepSeek: {e}")
        logger.error(f"Raw response (first 500 chars): {raw[:500]}")
        return AnalysisResponse(
            executive_summary="Analysis generated but could not be parsed. Please retry.",
            kalventis_insights="", gsk_insights="",
            recommendations=[], risk_indicators=[], opportunities=[],
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return AnalysisResponse(
            executive_summary=f"Analysis unavailable: {str(e)}",
            kalventis_insights="", gsk_insights="",
            recommendations=[], risk_indicators=[], opportunities=[],
        )


class DeepAnalysisRequest(BaseModel):
    brand_a_name: str
    brand_a_username: str
    brand_b_name: str
    brand_b_username: str
    brand_a: AccountSnapshot
    brand_b: AccountSnapshot
    comparison: dict
    top_terms: list[str] = []
    brand_a_top_terms: list[str] = []
    brand_b_top_terms: list[str] = []
    top_posts: list[dict] = []
    top_comments: list[dict] = []
    coverage: dict | None = None
    period: str = ""
    language: str = "en"


class DeepAnalysisResponse(BaseModel):
    executive_summary: str
    brand_a_insights: str
    brand_b_insights: str
    content_strategy: str
    competitive_analysis: str
    audience_insights: str
    sentiment_deep_dive: str
    risk_assessment: list[str]
    growth_opportunities: list[str]
    recommendations: list[str]
    sentiment_chart_insight: list[str] = []
    engagement_chart_insight: list[str] = []
    wordcloud_insight: list[str] = []
    sentiment_chart_b64: str | None = None
    engagement_chart_b64: str | None = None
    wordcloud_chart_b64: str | None = None
    brand_a_wordcloud_chart_b64: str | None = None
    brand_b_wordcloud_chart_b64: str | None = None


@app.post("/api/v1/monitoring/analysis")
async def deep_monitoring_analysis(request: DeepAnalysisRequest) -> DeepAnalysisResponse:
    if not DEEPSEEK_API_KEY:
        return DeepAnalysisResponse(
            executive_summary="DeepSeek API key not configured.",
            brand_a_insights="", brand_b_insights="",
            content_strategy="", competitive_analysis="",
            audience_insights="", sentiment_deep_dive="",
            risk_assessment=[], growth_opportunities=[], recommendations=[],
        )

    try:
        raw = ""

        a = request.brand_a
        b = request.brand_b
        a_total = a.sentiment.positive + a.sentiment.neutral + a.sentiment.negative
        b_total = b.sentiment.positive + b.sentiment.neutral + b.sentiment.negative
        a_pos_rate = f"{(a.sentiment.positive / max(1, a_total) * 100):.1f}%"
        b_pos_rate = f"{(b.sentiment.positive / max(1, b_total) * 100):.1f}%"
        a_neg_rate = f"{(a.sentiment.negative / max(1, a_total) * 100):.1f}%"
        b_neg_rate = f"{(b.sentiment.negative / max(1, b_total) * 100):.1f}%"

        comp = request.comparison
        top_posts_text = "\n".join(
            f"  [{p.get('side','')}] @{p.get('username','')}: \"{p.get('caption','')[:200]}\" — {p.get('engagement',0)} engagement, sentiment: {p.get('sentiment','Neutral')}"
            for p in request.top_posts[:10]
        ) if request.top_posts else "No post samples available"

        top_comments_text = "\n".join(
            f"  [{c.get('side','')}] @{c.get('ownerUsername','')}: \"{c.get('text','')[:200]}\""
            for c in request.top_comments[:8]
        ) if request.top_comments else "No comment samples available"

        lang_instr = "PENTING: Anda HARUS merespons dalam bahasa Indonesia saja. Jangan gunakan bahasa Inggris sama sekali." if request.language == "id" else "Respond in English only."
        analyst_role = (
            "Anda adalah konsultan senior komunikasi kesehatan masyarakat dengan pengalaman 15+ tahun dalam analisis media sosial, strategi konten vaksin, dan competitive intelligence di pasar Indonesia."
            if request.language == "id"
            else "You are a senior public health communications consultant with 15+ years of experience in social media analytics, vaccine content strategy, and competitive intelligence."
        )

        prompt = f"""{lang_instr}

{analyst_role}

Analyze the following comprehensive social media monitoring data for two brands. Provide deep, data-driven strategic analysis. Reference specific numbers from the data. Be candid about weaknesses and specific about opportunities.

=== BRAND A: {request.brand_a_name} (@{request.brand_a_username}) ===
- Followers: {a.followers:,}
- Posts scraped: {a.posts_scraped}
- Avg likes/post: {a.avg_likes:,}
- Avg comments/post: N/A (see engagement total)
- Total engagement: {a.total_engagement:,}
- Sentiment: {a.sentiment.positive} positive / {a.sentiment.neutral} neutral / {a.sentiment.negative} negative → {a_pos_rate} positive rate, {a_neg_rate} negative rate

=== BRAND B: {request.brand_b_name} (@{request.brand_b_username}) ===
- Followers: {b.followers:,}
- Posts scraped: {b.posts_scraped}
- Avg likes/post: {b.avg_likes:,}
- Total engagement: {b.total_engagement:,}
- Sentiment: {b.sentiment.positive} positive / {b.sentiment.neutral} neutral / {b.sentiment.negative} negative → {b_pos_rate} positive rate, {b_neg_rate} negative rate

=== COMPETITIVE COMPARISON ===
- Total engagement across both brands: {comp.get('engagementTotal', 0):,}
- {request.brand_a_name} engagement share: {comp.get('brandAEngagementShare', 0)}%
- {request.brand_b_name} engagement share: {comp.get('brandBEngagementShare', 0)}%
- {request.brand_a_name} post share: {comp.get('brandAPostShare', 0)}%
- {request.brand_b_name} post share: {comp.get('brandBPostShare', 0)}%

=== TOP TERMS (word cloud) ===
- Combined: {', '.join(request.top_terms[:15]) if request.top_terms else 'Not available'}
- {request.brand_a_name}: {', '.join(request.brand_a_top_terms[:15]) if request.brand_a_top_terms else 'Not available'}
- {request.brand_b_name}: {', '.join(request.brand_b_top_terms[:15]) if request.brand_b_top_terms else 'Not available'}

=== SAMPLE TOP POSTS (by engagement) ===
{top_posts_text}

=== SAMPLE AUDIENCE COMMENTS ===
{top_comments_text}

=== COVERAGE ASSESSMENT ===
- Status: {request.coverage.get('status','unknown') if request.coverage else 'unknown'}
- Score: {request.coverage.get('score','N/A')}%
- Posts with timestamps: {request.coverage.get('postsWithTimestamps',0) if request.coverage else 0}
- Note: {request.coverage.get('coverageNote','') if request.coverage else 'N/A'}

=== MONITORING PERIOD ===
{request.period or 'Recent scan window'}

=== CHART DATA FOR VISUAL ANALYSIS ===
The following data will be rendered into charts on the frontend. Use the raw numbers above when writing the chart-specific insight fields below:
- Sentiment Chart: positive/neutral/negative counts for both brands (see BRAND A and BRAND B sections above)
- Engagement Chart: followers, total engagement, and avg likes per post for both brands (see BRAND A and BRAND B sections above)

=== INSTRUCTIONS ===
Respond ONLY in valid JSON with exactly these keys. Every string field MUST contain substantive analysis (4-6 sentences minimum). Every list field MUST have 4-5 items. No markdown, no code fences.

{{
  "executive_summary": "4-5 sentence synthesis of the competitive landscape. State who leads on each key dimension (followers, engagement, sentiment), quantify the gap with exact numbers, explain what structural advantage or content pattern is driving it, and name the single highest-leverage insight a strategist should act on immediately.",
  "brand_a_insights": "4-6 sentences of deep analysis of {request.brand_a_name}. Cover: (1) content performance patterns and which themes drive the most engagement, (2) engagement quality (engagement-per-post vs raw volume), (3) their 2-3 strongest content pillars backed by the data, (4) their most visible weakness, and (5) one non-obvious opportunity hidden in the numbers.",
  "brand_b_insights": "4-6 sentences of deep analysis of {request.brand_b_name}. Same depth as above — cover their competitive differentiation, where they outperform and underperform, what the engagement data reveals about their content quality, and one strategic move that would materially close the gap with {request.brand_a_name}.",
  "content_strategy": "4-6 sentences of cross-brand content strategy. Name the specific themes (from the top terms and post samples) that drive the highest engagement across both brands. Identify which formats or caption styles correlate with higher engagement. Give one concrete recommendation for what each brand should do more of and one thing to stop.",
  "competitive_analysis": "4-6 sentences of head-to-head competitive positioning. State exact market share of voice percentages. Calculate and compare engagement efficiency (engagement ÷ followers) for both brands. Analyse content frequency vs content quality tradeoffs. Identify which conversation topics each brand owns and which are contested.",
  "audience_insights": "4-6 sentences on what the audience reveals through comments and engagement. Quote or paraphrase specific themes from the comment samples. Identify unmet information needs or repeated questions. Describe the language register (technical, conversational, emotional). Flag any sentiment patterns in comments that differ from post-level sentiment.",
  "sentiment_deep_dive": "4-6 sentences going beyond raw percentages. Explain what specific content types or topics appear to DRIVE positive vs negative sentiment. Identify whether negative sentiment is brand-specific or reflects market-wide attitudes (e.g. vaccine hesitancy). Describe the neutral cohort — are they fence-sitters or low-intent? Give one tactic to shift neutral to positive.",
  "risk_assessment": ["Specific, concrete risk 1 with severity level (High/Medium/Low)", "Risk 2 with severity", "Risk 3 with severity", "Risk 4 with severity", "Risk 5 with severity"],
  "growth_opportunities": ["Specific, immediately actionable opportunity 1 with expected impact", "Opportunity 2", "Opportunity 3", "Opportunity 4", "Opportunity 5"],
  "recommendations": ["Priority 1 — immediate action (next 30 days) with specific tactic", "Priority 2 — 30-60 days", "Priority 3 — 60-90 days", "Priority 4 — ongoing structural change", "Priority 5 — quick win achievable this week"],
  "sentiment_chart_insight": ["State the exact positive/neutral/negative counts for both brands and compute the positive-rate and negative-rate for each.", "Explain what the ratio between positive and negative sentiment reveals about audience trust and content resonance for each brand.", "Analyse the size of the neutral group — what does it signal about fence-sitters or low-intent followers?", "Flag the most notable pattern visible in the chart (e.g. one brand with no negatives, unusually high neutral, large gap between brands) and explain why it matters strategically.", "Explain whether the sentiment difference is driven by content quality, posting volume, topic choices, or audience composition.", "Give one concrete tactic the leading brand should protect to maintain its sentiment advantage, and one the trailing brand should adopt immediately."],
  "engagement_chart_insight": ["State the exact followers, total engagement, and avg likes values for both brands from the chart.", "Calculate the engagement-per-follower rate for each brand (total_engagement ÷ followers) and compare them — this reveals content quality independent of audience size.", "Explain what the gap between the followers differential and the engagement differential reveals — is the leader winning on reach, content resonance, or both?", "Analyse avg likes per post as a proxy for per-content quality and what the gap signals about each brand's ability to create high-performing individual posts.", "Identify whether the trailing brand's gap is primarily a reach problem (needs more followers) or a content quality problem (needs better posts per given audience).", "Name the single most actionable lever for the trailing brand: is it posting frequency, content theme, format, or audience growth?"],
  "wordcloud_insight": ["Identify the 3-4 dominant topic clusters visible in the combined word list and name the specific terms that anchor each cluster.", "Compare the {request.brand_a_name} and {request.brand_b_name} word lists: name the themes each brand appears to own and any overlap or contested territory.", "Explain what these clusters reveal about what the audience cares about most or what content consistently attracts engagement.", "Flag any term whose presence is surprising or whose absence is a strategic gap — what is the audience talking about that one brand is missing?", "Suggest one specific content angle or series concept for each brand based on its distinct word cloud."]
}}

Base every insight on the actual data provided. Reference specific numbers. If data is thin or coverage is partial, acknowledge the limitation and recommend a re-scan."""  # noqa: E501

        raw = await _deepseek_text(prompt, max_tokens=16384)

        data = _extract_json(raw)

        # Chart 2: modern styled charts shown to the frontend
        sentiment_b64 = _render_sentiment_chart_modern(
            a.sentiment, b.sentiment, request.brand_a_name, request.brand_b_name
        )
        engagement_b64 = _render_engagement_chart_modern(
            a, b, request.brand_a_name, request.brand_b_name
        )
        wordcloud_b64 = _render_wordcloud_chart(request.top_terms)
        brand_a_wordcloud_b64 = _render_wordcloud_chart(request.brand_a_top_terms)
        brand_b_wordcloud_b64 = _render_wordcloud_chart(request.brand_b_top_terms)
        return DeepAnalysisResponse(
            executive_summary=data.get("executive_summary", ""),
            brand_a_insights=data.get("brand_a_insights", ""),
            brand_b_insights=data.get("brand_b_insights", ""),
            content_strategy=data.get("content_strategy", ""),
            competitive_analysis=data.get("competitive_analysis", ""),
            audience_insights=data.get("audience_insights", ""),
            sentiment_deep_dive=data.get("sentiment_deep_dive", ""),
            risk_assessment=data.get("risk_assessment", []),
            growth_opportunities=data.get("growth_opportunities", []),
            recommendations=data.get("recommendations", []),
            sentiment_chart_insight=data.get("sentiment_chart_insight", []),
            engagement_chart_insight=data.get("engagement_chart_insight", []),
            wordcloud_insight=data.get("wordcloud_insight", []),
            sentiment_chart_b64=sentiment_b64,
            engagement_chart_b64=engagement_b64,
            wordcloud_chart_b64=wordcloud_b64,
            brand_a_wordcloud_chart_b64=brand_a_wordcloud_b64,
            brand_b_wordcloud_chart_b64=brand_b_wordcloud_b64,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Deep analysis JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {raw[:500] if raw else 'N/A'}")
        return DeepAnalysisResponse(
            executive_summary="Analysis generated but response could not be parsed. Please retry.",
            brand_a_insights="", brand_b_insights="",
            content_strategy="", competitive_analysis="",
            audience_insights="", sentiment_deep_dive="",
            risk_assessment=[], growth_opportunities=[], recommendations=[],
        )
    except Exception as e:
        logger.error(f"Deep analysis error: {e}")
        return DeepAnalysisResponse(
            executive_summary=f"Analysis unavailable: {str(e)}",
            brand_a_insights="", brand_b_insights="",
            content_strategy="", competitive_analysis="",
            audience_insights="", sentiment_deep_dive="",
            risk_assessment=[], growth_opportunities=[], recommendations=[],
        )


class KalventisPost(BaseModel):
    caption: str
    likes: int = 0
    comments: int = 0
    type: str = "image"


class KalventisAnalysisRequest(BaseModel):
    posts: list[KalventisPost]
    period: str = ""


class TopicItem(BaseModel):
    name: str
    mentions: int
    momentum: str
    summary: str


class KalventisAnalysisResponse(BaseModel):
    topics: list[TopicItem]
    content_summary: str
    patterns: list[str]
    recommendations: list[str]


@app.post("/api/v1/kalventis/overview-analysis")
async def kalventis_overview_analysis(request: KalventisAnalysisRequest) -> KalventisAnalysisResponse:
    if not DEEPSEEK_API_KEY:
        return KalventisAnalysisResponse(
            topics=[], content_summary="DEEPSEEK_API_KEY not configured.",
            patterns=[], recommendations=[],
        )

    try:
        posts_sample = request.posts[:15]
        posts_text = "\n".join(
            f"[{i+1}] {p.type}: {p.likes} likes, {p.comments} comments - {p.caption[:100]}"
            for i, p in enumerate(posts_sample)
        )
        period_label = request.period or 'recent window'

        prompt = f"""Analyze Instagram posts from @kenapaharusvaksin (Kalventis), an Indonesian vaccine education brand.

Posts ({period_label}):
{posts_text[:4000]}

Return ONLY valid JSON (no markdown):
{{
  "topics": [{{"name": "topic", "mentions": N, "momentum": "growing|steady|declining", "summary": "sentence"}}],
  "content_summary": "2-3 sentence summary of content strategy",
  "patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "recommendations": ["action 1", "action 2", "action 3", "action 4"]
}}"""

        raw = await _deepseek_text(prompt)
        data = _extract_json(raw)

        topics = [TopicItem(name=t.get("name",""), mentions=t.get("mentions",0), momentum=t.get("momentum","steady"), summary=t.get("summary","")) for t in data.get("topics",[])]

        return KalventisAnalysisResponse(
            topics=topics,
            content_summary=data.get("content_summary",""),
            patterns=data.get("patterns",[]),
            recommendations=data.get("recommendations",[]),
        )


    except json.JSONDecodeError as e:
        logger.error(f"Kalventis analysis JSON error: {e}")
        return KalventisAnalysisResponse(
            topics=[], content_summary="Analysis produced unparseable output. Please retry.",
            patterns=[], recommendations=[],
        )
    except Exception as e:
        logger.error(f"Kalventis analysis error: {e}")
        return KalventisAnalysisResponse(
            topics=[], content_summary=f"Analysis unavailable: {str(e)}",
            patterns=[], recommendations=[],
        )


# ---------------------------------------------------------------------------
# Full agentic analysis — competitive + topic + vision charts in one call
# ---------------------------------------------------------------------------

class AnalysisRecommendation(BaseModel):
    title: str
    message: str
    severity: str  # "high" | "medium" | "low" | "positive"


class ChartItem(BaseModel):
    title: str
    image_base64: str
    analysis: str


class FullAnalysisRequest(BaseModel):
    kalventis: AccountSnapshot
    gsk: AccountSnapshot
    top_topics: list[str] = []
    news_count: int = 0
    period: str = ""
    top_words: list[str] = []
    follower_ratio: float = 1.0
    post_ratio: float = 1.0
    posts: list[KalventisPost] = []


class FullAnalysisResponse(BaseModel):
    analysis_text: str
    key_findings: list[str]
    recommendations: list[AnalysisRecommendation]
    risk_indicators: list[str]
    opportunities: list[str]
    topics: list[TopicItem]
    content_summary: str
    patterns: list[str]
    content_recommendations: list[str]
    charts: list[ChartItem]
    created_at: str


async def _analyse_chart_vision(b64: str, title: str) -> str:
    """Extract text from a chart image via UnstructuredImageLoader, then analyze with DeepSeek."""
    import tempfile
    try:
        from langchain_community.document_loaders.image import UnstructuredImageLoader

        img_bytes = base64.b64decode(b64)

        def _load_image() -> str:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            try:
                loader = UnstructuredImageLoader(tmp_path)
                docs = loader.load()
                return "\n".join(d.page_content for d in docs)
            finally:
                os.unlink(tmp_path)

        chart_text = await asyncio.to_thread(_load_image)
        prompt = (
            f"You are analyzing a '{title}' chart for Kalventis, an Indonesian vaccine brand's "
            "social media analytics dashboard. The following text and labels were extracted "
            f"from the chart image:\n\n{chart_text}\n\n"
            "In 2-3 concise sentences, describe the key insight this chart reveals "
            "and what it means for the brand's content strategy."
        )
        return await _deepseek_text(prompt)
    except Exception as e:
        logger.error(f"Vision analysis failed for '{title}': {e}")
        return ""


@app.post("/api/v1/full-analysis")
async def full_analysis(request: FullAnalysisRequest) -> FullAnalysisResponse:
    if not DEEPSEEK_API_KEY:
        return FullAnalysisResponse(
            analysis_text="DeepSeek API key not configured.",
            key_findings=[], recommendations=[], risk_indicators=[], opportunities=[],
            topics=[], content_summary="", patterns=[], content_recommendations=[],
            charts=[], created_at=datetime.now(timezone.utc).isoformat(),
        )

    kv = request.kalventis
    gsk = request.gsk
    kv_total = kv.sentiment.positive + kv.sentiment.neutral + kv.sentiment.negative
    gsk_total = gsk.sentiment.positive + gsk.sentiment.neutral + gsk.sentiment.negative
    kv_pos_rate = f"{(kv.sentiment.positive / max(1, kv_total) * 100):.1f}%"
    gsk_pos_rate = f"{(gsk.sentiment.positive / max(1, gsk_total) * 100):.1f}%"

    # --- Generate charts ---
    sentiment_b64 = _render_sentiment_chart(kv.sentiment, gsk.sentiment, "Kalventis", "GSK")
    engagement_b64 = _render_engagement_chart(kv, gsk, "Kalventis", "GSK")
    wordcloud_b64 = _render_wordcloud_chart(request.top_words)

    # --- Vision analysis for each chart ---
    charts: list[ChartItem] = []
    charts.append(ChartItem(
        title="Sentiment Distribution",
        image_base64=sentiment_b64,
        analysis=await _analyse_chart_vision(sentiment_b64, "Sentiment Distribution"),
    ))
    charts.append(ChartItem(
        title="Engagement Metrics",
        image_base64=engagement_b64,
        analysis=await _analyse_chart_vision(engagement_b64, "Engagement Metrics"),
    ))
    if wordcloud_b64:
        charts.append(ChartItem(
            title="Word Cloud",
            image_base64=wordcloud_b64,
            analysis=await _analyse_chart_vision(wordcloud_b64, "Word Cloud"),
        ))

    # --- Competitive analysis ---
    competitive_prompt = f"""You are a senior social media analyst for Kalventis, an Indonesian vaccine awareness brand.
Analyze the following social listening data and respond ONLY in valid JSON (no markdown):

KALVENTIS (@kenapaharusvaksin) — Owned: Followers {kv.followers:,} | Posts {kv.posts_scraped} | Avg likes {kv.avg_likes} | Engagement {kv.total_engagement:,} | Sentiment {kv_pos_rate} positive
GSK (@ayokitavaksin) — Competitor: Followers {gsk.followers:,} | Posts {gsk.posts_scraped} | Avg likes {gsk.avg_likes} | Engagement {gsk.total_engagement:,} | Sentiment {gsk_pos_rate} positive
Follower ratio: {request.follower_ratio:.1f}x | Post ratio: {request.post_ratio:.1f}x
Topics: {', '.join(request.top_topics[:8]) or 'N/A'} | News monitored: {request.news_count}
Top terms: {', '.join(request.top_words[:10]) or 'N/A'} | Period: {request.period}

{{
  "analysis_text": "3-4 paragraph markdown narrative covering competitive landscape, performance highlights, and strategic outlook",
  "key_findings": ["concise finding 1", "finding 2", "finding 3", "finding 4"],
  "recommendations": [
    {{"title": "short title", "message": "actionable detail", "severity": "high|medium|low|positive"}},
    {{"title": "...", "message": "...", "severity": "..."}},
    {{"title": "...", "message": "...", "severity": "..."}},
    {{"title": "...", "message": "...", "severity": "..."}}
  ],
  "risk_indicators": ["specific risk 1", "risk 2", "risk 3"],
  "opportunities": ["growth opportunity 1", "opportunity 2", "opportunity 3"]
}}"""

    # --- Topic / content analysis ---
    posts_text = "\n".join(
        f"[{i+1}] {p.likes}L {p.comments}C — {p.caption[:120]}"
        for i, p in enumerate(request.posts[:15])
    ) or "No posts available"

    topic_prompt = f"""Analyze @kenapaharusvaksin (Kalventis) Instagram posts. Respond ONLY in valid JSON (no markdown):

{posts_text[:3500]}

{{
  "topics": [{{"name": "topic", "mentions": N, "momentum": "growing|steady|declining", "summary": "one sentence"}}],
  "content_summary": "2-3 sentence content strategy assessment",
  "patterns": ["engagement pattern 1", "pattern 2", "pattern 3", "pattern 4"],
  "content_recommendations": ["actionable recommendation 1", "2", "3", "4"]
}}"""

    comp_data: dict = {}
    topic_data: dict = {}

    try:
        comp_data = _extract_json(await _deepseek_text(competitive_prompt))
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        comp_data = {"analysis_text": f"Analysis unavailable: {e}", "key_findings": [],
                     "recommendations": [], "risk_indicators": [], "opportunities": []}

    try:
        topic_data = _extract_json(await _deepseek_text(topic_prompt))
    except Exception as e:
        logger.error(f"Topic analysis error: {e}")
        topic_data = {"topics": [], "content_summary": "", "patterns": [], "content_recommendations": []}

    topics = [
        TopicItem(name=t.get("name", ""), mentions=t.get("mentions", 0),
                  momentum=t.get("momentum", "steady"), summary=t.get("summary", ""))
        for t in topic_data.get("topics", [])
    ]

    raw_recs = comp_data.get("recommendations", [])
    recommendations = [
        AnalysisRecommendation(
            title=r.get("title", ""),
            message=r.get("message", ""),
            severity=r.get("severity", "low"),
        )
        for r in raw_recs if isinstance(r, dict)
    ]

    return FullAnalysisResponse(
        analysis_text=comp_data.get("analysis_text", ""),
        key_findings=comp_data.get("key_findings", []),
        recommendations=recommendations,
        risk_indicators=comp_data.get("risk_indicators", []),
        opportunities=comp_data.get("opportunities", []),
        topics=topics,
        content_summary=topic_data.get("content_summary", ""),
        patterns=topic_data.get("patterns", []),
        content_recommendations=topic_data.get("content_recommendations", []),
        charts=charts,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Chat endpoint — conversational Q&A with social listening context
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class SocialListeningContext(BaseModel):
    kalventis_followers: int = 0
    kalventis_posts: int = 0
    kalventis_avg_likes: int = 0
    kalventis_total_engagement: int = 0
    kalventis_sentiment: dict = {}
    gsk_followers: int = 0
    gsk_posts: int = 0
    gsk_avg_likes: int = 0
    gsk_total_engagement: int = 0
    gsk_sentiment: dict = {}
    top_topics: list[str] = []
    top_words: list[str] = []
    news_count: int = 0
    follower_ratio: float = 1.0
    post_ratio: float = 1.0
    period: str = ""


class SocialChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: SocialListeningContext | None = None


class SocialChatResponse(BaseModel):
    response: str


@app.post("/api/v1/chat")
async def social_chat(request: SocialChatRequest) -> SocialChatResponse:
    if not DEEPSEEK_API_KEY:
        return SocialChatResponse(response="DeepSeek API key not configured.")

    ctx = request.context
    if ctx:
        kv_sent = ctx.kalventis_sentiment
        gsk_sent = ctx.gsk_sentiment
        context_block = (
            "You are an expert social media analyst assistant for Kalventis, "
            "an Indonesian vaccine awareness brand (@kenapaharusvaksin).\n"
            "When answering, prioritize public knowledge first: industry benchmarks, platform best practices, "
            "competitor landscape, and external research. Then use the internal dashboard data below to "
            "validate, contrast, or add specifics — the digital marketing team already knows their own numbers, "
            "so lead with external insight and use internal data to support or contextualize.\n\n"
            "=== INTERNAL DASHBOARD DATA (use to support, not lead) ===\n"
            "=== KALVENTIS (@kenapaharusvaksin) ===\n"
            f"Followers: {ctx.kalventis_followers:,}\n"
            f"Posts scraped: {ctx.kalventis_posts}\n"
            f"Avg likes/post: {ctx.kalventis_avg_likes}\n"
            f"Total engagement: {ctx.kalventis_total_engagement:,}\n"
            f"Sentiment: {kv_sent.get('positive', 0)} positive / "
            f"{kv_sent.get('neutral', 0)} neutral / {kv_sent.get('negative', 0)} negative\n\n"
            "=== GSK COMPETITOR (@ayokitavaksin) ===\n"
            f"Followers: {ctx.gsk_followers:,}\n"
            f"Posts scraped: {ctx.gsk_posts}\n"
            f"Avg likes/post: {ctx.gsk_avg_likes}\n"
            f"Total engagement: {ctx.gsk_total_engagement:,}\n"
            f"Sentiment: {gsk_sent.get('positive', 0)} positive / "
            f"{gsk_sent.get('neutral', 0)} neutral / {gsk_sent.get('negative', 0)} negative\n\n"
            "=== COMPETITIVE POSITION ===\n"
            f"Follower ratio: Kalventis is {ctx.follower_ratio:.1f}x larger\n"
            f"Post ratio: Kalventis is {ctx.post_ratio:.1f}x more active\n"
            f"Period: {ctx.period}\n\n"
            "=== TOPICS THE TEAM ALREADY TRACKS (the known baseline — do NOT just repeat these) ===\n"
            f"Already-tracked topics: {', '.join(ctx.top_topics[:8]) or 'None'}\n"
            f"Already-tracked words: {', '.join(ctx.top_words[:12]) or 'None'}\n"
            f"News articles monitored: {ctx.news_count}\n\n"
            "The digital marketing team already knows the topics listed above. Your main job is the OPPOSITE: "
            "surface popular and emerging topics from the wider public conversation that are NOT already in "
            "that list — adjacent themes, trending angles, audience questions, and competitor topics the team "
            "may be missing. Treat the internal list only as the baseline of what they already cover. "
            "Answer concisely and practically, use bullet points where helpful, and be specific with numbers "
            "when relevant — but lead with the NEW external topics worth their attention."
        )
        context_data = {
            "brand_a_name": "Kalventis",
            "brand_b_name": "GSK",
            "brand_a": {
                "followers": ctx.kalventis_followers,
                "posts_scraped": ctx.kalventis_posts,
                "avg_likes": ctx.kalventis_avg_likes,
                "total_engagement": ctx.kalventis_total_engagement,
                "sentiment": kv_sent,
            },
            "brand_b": {
                "followers": ctx.gsk_followers,
                "posts_scraped": ctx.gsk_posts,
                "avg_likes": ctx.gsk_avg_likes,
                "total_engagement": ctx.gsk_total_engagement,
                "sentiment": gsk_sent,
            },
        }
        context_meta = {
            "period": ctx.period,
            "brand_a_name": "Kalventis",
            "brand_b_name": "GSK",
        }
    else:
        context_block = (
            "You are a social media analyst assistant for Kalventis, an Indonesian vaccine "
            "awareness brand. Answer questions about social media strategy, vaccine content, "
            "and competitive analysis."
        )
        context_data = {}
        context_meta = {"period": "current period", "brand_a_name": "Kalventis", "brand_b_name": "GSK"}

    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        response_text = await asyncio.to_thread(
            run_agentic_chat,
            message=request.message,
            history=history,
            context_block=context_block,
            context_data=context_data,
            context_meta=context_meta,
        )
        return SocialChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Agentic chat error: {e}")
        return SocialChatResponse(response="Sorry, I couldn't process your question. Please try again.")


# ---------------------------------------------------------------------------
# Monitoring chat — conversational Q&A grounded in a monitoring scan result
# ---------------------------------------------------------------------------

class MonitoringChatContext(BaseModel):
    brand_a_name: str = ""
    brand_a_username: str = ""
    brand_b_name: str = ""
    brand_b_username: str = ""
    brand_a: AccountSnapshot = AccountSnapshot()
    brand_b: AccountSnapshot = AccountSnapshot()
    comparison: dict = {}
    top_terms: list[str] = []
    brand_a_top_terms: list[str] = []
    brand_b_top_terms: list[str] = []
    top_posts: list[dict] = []
    top_comments: list[dict] = []
    coverage: dict = {}
    period: str = ""


class MonitoringChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: MonitoringChatContext | None = None


@app.post("/api/v1/monitoring/chat")
async def monitoring_chat(request: MonitoringChatRequest) -> SocialChatResponse:
    if not DEEPSEEK_API_KEY:
        return SocialChatResponse(response="DeepSeek API key not configured.")

    ctx = request.context
    if ctx:
        a = ctx.brand_a
        b = ctx.brand_b
        comp = ctx.comparison
        a_total = max(1, a.sentiment.positive + a.sentiment.neutral + a.sentiment.negative)
        b_total = max(1, b.sentiment.positive + b.sentiment.neutral + b.sentiment.negative)

        posts_text = "\n".join(
            f"  [{p.get('side','')}] @{p.get('username','')}: "
            f"\"{str(p.get('caption',''))[:150]}\" — {p.get('engagement',0)} eng, {p.get('sentiment','Neutral')}"
            for p in ctx.top_posts[:8]
        ) or "  No post samples available."

        comments_text = "\n".join(
            f"  [{c.get('side','')}] @{c.get('ownerUsername','')}: \"{str(c.get('text',''))[:150]}\""
            for c in ctx.top_comments[:6]
        ) or "  No comment samples available."

        context_block = (
            "You are an expert social media competitive intelligence analyst.\n"
            "The digital marketing team already knows the themes and numbers in their own monitoring scan "
            "(shown below). Your primary value is to surface popular and emerging topics from the wider public "
            "conversation that are NOT already in their tracked themes — trending angles, adjacent subjects, "
            "audience questions, and competitor topics they may be missing. Lead with these NEW external topics "
            "and industry benchmarks first, then use the monitoring scan data below only as the baseline to "
            "contrast against and to show where the gaps and opportunities are.\n\n"
            f"=== {ctx.brand_a_name} (@{ctx.brand_a_username}) ===\n"
            f"Followers: {a.followers:,}\n"
            f"Posts scraped: {a.posts_scraped}\n"
            f"Avg likes / post: {a.avg_likes:,}\n"
            f"Total engagement: {a.total_engagement:,}\n"
            f"Sentiment: {a.sentiment.positive} positive ({a.sentiment.positive/a_total*100:.0f}%) / "
            f"{a.sentiment.neutral} neutral ({a.sentiment.neutral/a_total*100:.0f}%) / "
            f"{a.sentiment.negative} negative ({a.sentiment.negative/a_total*100:.0f}%)\n\n"
            f"=== {ctx.brand_b_name} (@{ctx.brand_b_username}) ===\n"
            f"Followers: {b.followers:,}\n"
            f"Posts scraped: {b.posts_scraped}\n"
            f"Avg likes / post: {b.avg_likes:,}\n"
            f"Total engagement: {b.total_engagement:,}\n"
            f"Sentiment: {b.sentiment.positive} positive ({b.sentiment.positive/b_total*100:.0f}%) / "
            f"{b.sentiment.neutral} neutral ({b.sentiment.neutral/b_total*100:.0f}%) / "
            f"{b.sentiment.negative} negative ({b.sentiment.negative/b_total*100:.0f}%)\n\n"
            "=== COMPETITIVE COMPARISON ===\n"
            f"Total engagement (both brands): {comp.get('engagementTotal', 0):,}\n"
            f"{ctx.brand_a_name} engagement share: {comp.get('brandAEngagementShare', 0)}%\n"
            f"{ctx.brand_b_name} engagement share: {comp.get('brandBEngagementShare', 0)}%\n"
            f"{ctx.brand_a_name} post share: {comp.get('brandAPostShare', 0)}%\n"
            f"{ctx.brand_b_name} post share: {comp.get('brandBPostShare', 0)}%\n\n"
            "=== CONTENT THEMES THE TEAM ALREADY TRACKS (known baseline — find topics BEYOND these) ===\n"
            f"Combined: {', '.join(ctx.top_terms[:15]) or 'Not available'}\n"
            f"{ctx.brand_a_name}: {', '.join(ctx.brand_a_top_terms[:15]) or 'Not available'}\n"
            f"{ctx.brand_b_name}: {', '.join(ctx.brand_b_top_terms[:15]) or 'Not available'}\n\n"
            "=== TOP POSTS (by engagement) ===\n"
            f"{posts_text}\n\n"
            "=== AUDIENCE COMMENTS SAMPLE ===\n"
            f"{comments_text}\n\n"
            "=== COVERAGE ===\n"
            f"Status: {ctx.coverage.get('status', 'unknown')} · Score: {ctx.coverage.get('score', 'N/A')}%\n"
            f"Note: {ctx.coverage.get('coverageNote', '')}\n\n"
            f"Period: {ctx.period or 'Recent scan'}"
        )
        context_data = {
            "brand_a_name": ctx.brand_a_name or "Brand A",
            "brand_b_name": ctx.brand_b_name or "Brand B",
            "brand_a": {
                "followers": a.followers,
                "posts_scraped": a.posts_scraped,
                "avg_likes": a.avg_likes,
                "total_engagement": a.total_engagement,
                "sentiment": a.sentiment,
            },
            "brand_b": {
                "followers": b.followers,
                "posts_scraped": b.posts_scraped,
                "avg_likes": b.avg_likes,
                "total_engagement": b.total_engagement,
                "sentiment": b.sentiment,
            },
        }
        context_meta = {
            "period": ctx.period or "Recent scan",
            "brand_a_name": ctx.brand_a_name or "Brand A",
            "brand_b_name": ctx.brand_b_name or "Brand B",
        }
    else:
        context_block = (
            "You are a social media competitive intelligence analyst. "
            "Answer questions about social media strategy and competitive analysis."
        )
        context_data = {}
        context_meta = {"period": "current period", "brand_a_name": "Brand A", "brand_b_name": "Brand B"}

    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        response_text = await asyncio.to_thread(
            run_agentic_chat,
            message=request.message,
            history=history,
            context_block=context_block,
            context_data=context_data,
            context_meta=context_meta,
        )
        return SocialChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Agentic monitoring chat error: {e}")
        return SocialChatResponse(response="Sorry, I couldn't process your question. Please try again.")


# ---------------------------------------------------------------------------
# IndoBERT batch sentiment classification endpoint
# ---------------------------------------------------------------------------

SentimentText = Annotated[str, Field(max_length=_SENTIMENT_MAX_CHARS)]


class SentimentRequest(BaseModel):
    texts: list[SentimentText] = Field(default_factory=list, max_length=_SENTIMENT_MAX_BATCH)


class SentimentResult(BaseModel):
    label: str   # "Positive" | "Neutral" | "Negative"
    score: float


class SentimentResponse(BaseModel):
    results: list[SentimentResult]


@app.post("/api/v1/sentiment")
async def classify_sentiment(request: SentimentRequest) -> SentimentResponse:
    if not request.texts:
        return SentimentResponse(results=[])

    try:
        pipe = await _get_sentiment_pipeline()

        def _run_inference() -> list[dict]:
            out: list[dict | None] = [None] * len(request.texts)
            indexed_texts: list[tuple[int, str]] = []
            for index, text in enumerate(request.texts):
                if not text or not text.strip():
                    out[index] = {"label": "Neutral", "score": 1.0}
                    continue
                indexed_texts.append((index, text[:_SENTIMENT_INFERENCE_CHARS]))

            if indexed_texts:
                results = pipe(
                    [text for _, text in indexed_texts],
                    batch_size=_SENTIMENT_BATCH_SIZE,
                    truncation=True,
                    max_length=_SENTIMENT_INFERENCE_CHARS,
                )
                for (index, _), result in zip(indexed_texts, results):
                    out[index] = {
                        "label": _LABEL_MAP.get(result["label"], "Neutral"),
                        "score": float(result["score"]),
                    }
            return [item or {"label": "Neutral", "score": 0.0} for item in out]

        raw = await asyncio.to_thread(_run_inference)
        return SentimentResponse(results=[SentimentResult(**r) for r in raw])

    except Exception as e:
        logger.error(f"Sentiment classification error: {e}")
        raise HTTPException(status_code=503, detail="Sentiment model unavailable") from e
