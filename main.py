"""Social Listening AI Analysis API — Python backend for LLM-powered insights."""

import json
import logging
import os
import re

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

        # Extract JSON from markdown code fences or free text.
        # Handles: ```json{...}```, ```{...}```, bare {...}, {...} with trailing text.
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", raw, 0)
        raw = json_match.group(0)

        # Remove trailing commas before closing brackets (common LLM mistake)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)

        # Strip BOM or invisible chars that break json.loads
        raw = raw.strip().lstrip('\ufeff')

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


@app.post("/api/v1/monitoring/analysis")
async def deep_monitoring_analysis(request: DeepAnalysisRequest) -> DeepAnalysisResponse:
    if not GEMINI_API_KEY:
        return DeepAnalysisResponse(
            executive_summary="Gemini API key not configured.",
            brand_a_insights="", brand_b_insights="",
            content_strategy="", competitive_analysis="",
            audience_insights="", sentiment_deep_dive="",
            risk_assessment=[], growth_opportunities=[], recommendations=[],
        )

    try:
        raw = ""
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

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
{', '.join(request.top_terms[:15]) if request.top_terms else 'Not available'}

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

=== INSTRUCTIONS ===
Respond ONLY in valid JSON with exactly these keys. Every string field MUST contain substantive analysis (2-5 sentences). Every list field MUST have 3-5 items. No markdown, no code fences.

{{
  "executive_summary": "2-3 sentence synthesis of the competitive landscape. Who leads, on what dimensions, and what is the single most important strategic insight from this data.",
  "brand_a_insights": "Deep analysis of {request.brand_a_name}. Content performance patterns, engagement quality (not just volume), what content types/themes resonate. Identify their 2-3 strongest content pillars and any weaknesses visible in the data.",
  "brand_b_insights": "Deep analysis of {request.brand_b_name}. Same depth as above. What is their competitive differentiation? Where do they outperform and underperform?",
  "content_strategy": "Cross-brand content strategy analysis. What content themes drive highest engagement across both brands? What formats, caption styles, or post types correlate with higher engagement? What should each brand do more/less of based on the data?",
  "competitive_analysis": "Head-to-head competitive positioning. Market share of voice, engagement efficiency (engagement per post vs follower count), content frequency vs quality tradeoffs. Who owns which conversation themes?",
  "audience_insights": "What the audience reveals through comments and engagement patterns. What questions, concerns, or topics do they raise? What language/terminology do they use? Are there unmet information needs?",
  "sentiment_deep_dive": "Beyond positive/neutral/negative percentages: what DRIVES sentiment? What topics or content types correlate with positive vs negative responses? Are there sentiment patterns across the two brands that suggest market-wide attitudes vs brand-specific reactions?",
  "risk_assessment": ["Specific, concrete risk 1 with severity", "Risk 2", "Risk 3", "Risk 4"],
  "growth_opportunities": ["Specific, actionable opportunity 1", "Opportunity 2", "Opportunity 3", "Opportunity 4"],
  "recommendations": ["Prioritized action 1 (immediate, next 30 days)", "Action 2 (30-60 days)", "Action 3 (60-90 days)", "Action 4 (ongoing)", "Action 5 (quick win)"]
}}

Base every insight on the actual data provided. Reference specific numbers. If data is thin or coverage is partial, acknowledge the limitation and recommend a re-scan."""  # noqa: E501

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 2048},
        )
        raw = response.text.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", raw, 0)
        raw = json_match.group(0)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        raw = raw.strip().lstrip('\ufeff')

        data = json.loads(raw)

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
