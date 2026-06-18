"""News intelligence + viral-angle engine for Social Listening.

Two capabilities, both LLM-powered and shared by the agentic chat and the
``/api/v1/analysis`` endpoint:

Part 2 — Deeper LLM intake (credible news gathering)
    expand_queries → search_news (Tavily topic=news) → extract_facts
    → corroborate → ranked, credibility-scored NewsBrief.

Part 3 — Viral-angle engine
    generate_viral_angles fuses the internal scraped sentiment with the
    external news brief, scores each angle on virality drivers, and every
    angle is run through brand_safety_review — a pharma/health guardrail
    that flags medical claims, misinformation, and crisis-newsjacking
    before a marketer ever sees them.

The module is self-contained: its own DeepSeek client, its own JSON
extractor, and graceful degradation when Tavily / the API key is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import OpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


# ─── Source credibility tiers (Indonesian vaccine/health focus) ───────────────
# Tier 1 = authoritative (govt, top wire/outlets, medical bodies, journals)
# Tier 2 = mainstream but mid-tier
# Tier 3 = everything else → trend signal only, low fact-trust
_TIER_1 = {
    # Indonesian government & medical authorities
    "kemkes.go.id", "kemenkes.go.id", "covid19.go.id", "pom.go.id",
    "litbang.kemkes.go.id", "sehatnegeriku.kemkes.go.id",
    "idai.or.id", "idionline.org",
    # Indonesian top outlets / wire
    "antaranews.com", "kompas.com", "kompas.id", "tempo.co",
    "cnnindonesia.com", "detik.com",
    # Global wire / journals / health
    "reuters.com", "apnews.com", "who.int", "nature.com",
    "thelancet.com", "nejm.org", "bbc.com", "cdc.gov",
}
_TIER_2 = {
    "tribunnews.com", "liputan6.com", "kumparan.com", "suara.com",
    "republika.co.id", "sindonews.com", "medcom.id", "cnbcindonesia.com",
    "jpnn.com", "tirto.id", "katadata.co.id", "healthline.com",
    "medicalnewstoday.com", "theconversation.com",
}
_TIER_SCORE = {1: 1.0, 2: 0.6, 3: 0.3}


def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return (m.group(1) if m else "").lower()


def _tier_for(url: str) -> int:
    d = _domain(url)
    if any(d == t or d.endswith("." + t) for t in _TIER_1):
        return 1
    if any(d == t or d.endswith("." + t) for t in _TIER_2):
        return 2
    return 3


def _recency_weight(published: str | None) -> float:
    """0..1 freshness weight; ~7-day half-life. Unknown dates → neutral 0.5."""
    if not published:
        return 0.5
    dt = None
    for parser in (
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        parsedate_to_datetime,
    ):
        try:
            dt = parser(published)
            break
        except (ValueError, TypeError):
            continue
    if dt is None:
        return 0.5
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    return 0.5 ** (age_days / 7.0)  # 1.0 today, 0.5 at 7d, 0.25 at 14d


# ─── LLM helpers ──────────────────────────────────────────────────────────────

def _client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=120.0)


def _llm(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    resp = _client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(raw: str):
    """Extract the first balanced JSON object/array from an LLM response."""
    if not raw:
        return None
    cleaned = raw.strip().lstrip("﻿")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).replace("```", "")
    start = min(
        [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1],
        default=-1,
    )
    if start == -1:
        return None
    open_ch = cleaned[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ─── Part 2: Deeper LLM intake ────────────────────────────────────────────────

def expand_queries(topic: str, market: str = "Indonesia", n: int = 5) -> list[str]:
    """Turn one topic into N diverse search queries (ID + EN) for wide recall."""
    prompt = (
        f"You are a research strategist. Expand this topic into {n} diverse, "
        f"high-recall NEWS search queries for the {market} market.\n\n"
        f"Topic: \"{topic}\"\n\n"
        "Rules:\n"
        "- Mix Indonesian (Bahasa) and English queries.\n"
        "- Vary the angle: official/regulatory, public reaction, expert/medical, "
        "controversy/debate, and latest developments.\n"
        "- Each query 4-9 words, optimized for a news search engine.\n"
        'Return ONLY a JSON array of strings, e.g. ["q1","q2",...].'
    )
    try:
        data = _extract_json(_llm(prompt, max_tokens=512, temperature=0.4))
        queries = [q.strip() for q in data if isinstance(q, str) and q.strip()] if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001
        logger.warning("[news] expand_queries failed: %s", e)
        queries = []
    if topic not in queries:
        queries.insert(0, topic)
    return queries[:n]


def search_news(queries: list[str], days: int = 7, max_per_query: int = 6) -> list[dict]:
    """Run Tavily news search across queries; dedupe by URL. No LLM."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("[news] TAVILY_API_KEY not set — skipping news search")
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("[news] tavily-python not installed")
        return []

    client = TavilyClient(api_key=api_key)
    seen: dict[str, dict] = {}
    for q in queries:
        try:
            resp = client.search(
                query=q,
                topic="news",
                days=days,
                search_depth="advanced",
                max_results=max_per_query,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[news] search failed for %r: %s", q, e)
            continue
        for r in resp.get("results", []):
            url = r.get("url", "")
            if not url or url in seen:
                continue
            tier = _tier_for(url)
            rec = _recency_weight(r.get("published_date"))
            seen[url] = {
                "title": r.get("title") or url,
                "url": url,
                "domain": _domain(url),
                "content": (r.get("content") or "")[:700],
                "published_date": r.get("published_date"),
                "tier": tier,
                "credibility": _TIER_SCORE[tier],
                "recency": round(rec, 3),
                # rank blends source authority (60%) and freshness (40%)
                "rank": round(0.6 * _TIER_SCORE[tier] + 0.4 * rec, 3),
            }
    articles = sorted(seen.values(), key=lambda a: a["rank"], reverse=True)
    logger.info("[news] %d unique articles from %d queries", len(articles), len(queries))
    return articles


def extract_and_corroborate(topic: str, articles: list[dict], max_articles: int = 10) -> dict:
    """Map-reduce in one LLM call: structured facts + cross-source corroboration.

    Returns {verified_facts:[{claim, sources:[idx], source_count, tier_min}],
             unverified_claims:[{claim, source}], summary, sentiment}.
    A claim is 'verified' when ≥2 independent sources report it.
    """
    if not articles:
        return {"verified_facts": [], "unverified_claims": [], "summary": "", "sentiment": "Neutral"}

    top = articles[:max_articles]
    block = "\n\n".join(
        f"[{i}] {a['title']} (source: {a['domain']}, tier {a['tier']}, published {a.get('published_date') or 'unknown'})\n{a['content']}"
        for i, a in enumerate(top)
    )
    prompt = (
        "You are a fact-checking news analyst. Below are recent news snippets about a topic, "
        "each tagged with a source-credibility tier (1 = most authoritative).\n\n"
        f"TOPIC: {topic}\n\nARTICLES:\n{block}\n\n"
        "Do the following and return ONLY JSON:\n"
        "1. Extract atomic factual CLAIMS from the articles.\n"
        "2. A claim is VERIFIED only if it appears in 2+ independent sources (different [index]). "
        "Otherwise it is UNVERIFIED.\n"
        "3. Write a neutral 2-3 sentence summary of what is credibly happening.\n"
        "4. Give overall public sentiment toward the topic: Positive, Neutral, or Negative.\n\n"
        "{\n"
        '  "summary": "...",\n'
        '  "sentiment": "Positive|Neutral|Negative",\n'
        '  "verified_facts": [{"claim": "...", "sources": [0,2], "source_count": 2, "tier_min": 1}],\n'
        '  "unverified_claims": [{"claim": "...", "source": 3}]\n'
        "}"
    )
    data = _extract_json(_llm(prompt, max_tokens=4096, temperature=0.1)) or {}
    return {
        "summary": data.get("summary", ""),
        "sentiment": data.get("sentiment", "Neutral"),
        "verified_facts": data.get("verified_facts", []) or [],
        "unverified_claims": data.get("unverified_claims", []) or [],
    }


def gather_news(topic: str, days: int = 7, max_articles: int = 10, n_queries: int = 5) -> dict:
    """Full credible-news pipeline → NewsBrief dict."""
    if not topic or not topic.strip():
        return {"topic": topic, "articles": [], "verified_facts": [], "unverified_claims": [],
                "summary": "", "sentiment": "Neutral", "credible_source_count": 0}
    queries = expand_queries(topic, n=n_queries)
    articles = search_news(queries, days=days)
    analysis = extract_and_corroborate(topic, articles, max_articles=max_articles)
    return {
        "topic": topic,
        "queries": queries,
        "articles": articles[:max_articles],
        "credible_source_count": sum(1 for a in articles if a["tier"] <= 2),
        **analysis,
    }


def brief_to_prompt_block(brief: dict) -> str:
    """Render a NewsBrief as a compact, citation-friendly text block for an LLM."""
    if not brief or (not brief.get("articles") and not brief.get("summary")):
        return ""
    lines = ["\n\n=== CREDIBLE NEWS BRIEF ===",
             f"Topic: {brief.get('topic', '')}",
             f"Public sentiment: {brief.get('sentiment', 'Neutral')} | "
             f"Credible sources: {brief.get('credible_source_count', 0)}"]
    if brief.get("summary"):
        lines.append(f"Summary: {brief['summary']}")
    if brief.get("verified_facts"):
        lines.append("VERIFIED FACTS (corroborated by 2+ sources):")
        for f in brief["verified_facts"][:8]:
            lines.append(f"  ✓ {f.get('claim','')} ({f.get('source_count',0)} sources)")
    if brief.get("unverified_claims"):
        lines.append("UNVERIFIED (single-source — treat with caution):")
        for c in brief["unverified_claims"][:5]:
            lines.append(f"  ? {c.get('claim','')}")
    if brief.get("articles"):
        lines.append("SOURCES:")
        for a in brief["articles"][:8]:
            lines.append(f"  🌐 [{a['title']}]({a['url']}) — tier {a['tier']}")
    return "\n".join(lines)


# ─── Part 3: Viral-angle engine ───────────────────────────────────────────────

def generate_viral_angles(
    topic: str,
    news_brief: dict | None = None,
    internal_context: str = "",
    brand: str = "Kalventis (@kenapaharusvaksin)",
    n: int = 4,
) -> list[dict]:
    """Fuse internal sentiment + credible news into scored, platform-ready angles."""
    news_block = brief_to_prompt_block(news_brief or {})
    prompt = (
        "You are a viral content strategist for a digital marketing team. "
        f"The brand is {brand}, an Indonesian public-health / vaccine-awareness brand.\n\n"
        f"TOPIC: {topic}\n"
        f"{internal_context}\n"
        f"{news_block}\n\n"
        f"Generate {n} distinct VIRAL content angles that fuse the credible news with what the "
        "brand's audience already engages with. ONLY use facts from the brief — never invent claims.\n\n"
        "For each angle score these virality drivers 1-10: emotion, novelty, timeliness, "
        "relatability, shareability. Compute virality_score = average × 10 (0-100).\n\n"
        "Return ONLY a JSON array:\n"
        "[{\n"
        '  "angle": "the content idea in one sentence",\n'
        '  "platform": "TikTok|Instagram Reels|Instagram Carousel|X/Twitter Thread",\n'
        '  "format": "short concrete format",\n'
        '  "hook": "the exact first 3 seconds / opening line (Bahasa Indonesia)",\n'
        '  "why_it_works": "1 sentence tied to the data/news",\n'
        '  "scores": {"emotion":0,"novelty":0,"timeliness":0,"relatability":0,"shareability":0},\n'
        '  "virality_score": 0,\n'
        '  "grounded_in": "which verified fact or metric this uses"\n'
        "}]"
    )
    angles = _extract_json(_llm(prompt, max_tokens=4096, temperature=0.6))
    if not isinstance(angles, list):
        return []
    angles.sort(key=lambda a: a.get("virality_score", 0), reverse=True)
    return angles


def brand_safety_review(angles: list[dict], domain: str = "vaccine/public-health (Indonesia)") -> list[dict]:
    """Pharma/health guardrail — annotate each angle with a compliance verdict.

    Adds a 'safety' object: {verdict: safe|caution|block, flags:[...], fix: "..."}.
    Flags unapproved medical/efficacy claims, misinformation, fear-mongering on
    health crises, anti-vax amplification, and unsafe newsjacking.
    """
    if not angles:
        return angles
    block = "\n".join(
        f"[{i}] platform={a.get('platform','')} | angle={a.get('angle','')} | hook={a.get('hook','')}"
        for i, a in enumerate(angles)
    )
    prompt = (
        f"You are a regulatory/brand-safety reviewer for {domain} marketing "
        "(rules similar to BPOM/Kemenkes — no unapproved medical claims, no health misinformation, "
        "no exploiting a health crisis, no anti-vaccine amplification).\n\n"
        f"Review each content angle:\n{block}\n\n"
        "For each index return a verdict:\n"
        "- safe: publish as-is\n"
        "- caution: publishable only with the suggested fix\n"
        "- block: do not publish\n\n"
        "Return ONLY JSON array aligned by index:\n"
        '[{"index":0,"verdict":"safe|caution|block","flags":["..."],"fix":"one-line fix or empty"}]'
    )
    verdicts = _extract_json(_llm(prompt, max_tokens=2048, temperature=0.1))
    by_idx = {v.get("index"): v for v in verdicts} if isinstance(verdicts, list) else {}
    for i, a in enumerate(angles):
        v = by_idx.get(i, {})
        a["safety"] = {
            "verdict": v.get("verdict", "caution"),
            "flags": v.get("flags", []) or [],
            "fix": v.get("fix", ""),
        }
    return angles


def viral_pipeline(
    topic: str,
    news_brief: dict | None = None,
    internal_context: str = "",
    brand: str = "Kalventis (@kenapaharusvaksin)",
    n: int = 4,
) -> list[dict]:
    """Generate angles then run them through the safety guardrail."""
    angles = generate_viral_angles(topic, news_brief, internal_context, brand, n)
    return brand_safety_review(angles)
