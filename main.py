"""Social Listening AI Analysis API — Python backend for LLM-powered insights."""

import json
import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Social Listening AI Analysis API", version="1.0.0")

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analysis")
async def generate_analysis(request: AnalysisRequest) -> AnalysisResponse:
    if not GEMINI_API_KEY:
        return AnalysisResponse(
            executive_summary="Gemini API key not configured. Add GEMINI_API_KEY to .env to enable AI analysis.",
            kalventis_insights="Configure GEMINI_API_KEY to enable Kalventis insights.",
            gsk_insights="Configure GEMINI_API_KEY to enable GSK competitive insights.",
            recommendations=["Add GEMINI_API_KEY to D:\\fastapi_all\\python-social-listening\\.env"],
            risk_indicators=[],
            opportunities=[],
        )

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

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

        response = model.generate_content(prompt)
        raw = response.text.strip()

        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        return AnalysisResponse(
            executive_summary=data.get("executive_summary", ""),
            kalventis_insights=data.get("kalventis_insights", ""),
            gsk_insights=data.get("gsk_insights", ""),
            recommendations=data.get("recommendations", []),
            risk_indicators=data.get("risk_indicators", []),
            opportunities=data.get("opportunities", []),
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from Gemini: {e}")
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
