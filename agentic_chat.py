"""Agentic chat for Social Listening — LangGraph routing + LangChain Python Agent for visualization."""

import base64
import io
import logging
import os
import re
from typing import Optional, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import news_intelligence as ni

load_dotenv()

logger = logging.getLogger(__name__)


# ─── State ───────────────────────────────────────────────────────────────────

class AgenticChatState(TypedDict):
    message: str
    history: list
    context_block: str
    context_data: dict
    context_meta: dict
    needs_web_search: bool
    needs_visualization: bool
    needs_news: bool
    needs_viral: bool
    topic: str
    web_query: str
    web_results: list
    news_brief: dict
    viral_angles: list
    viz_b64: Optional[str]
    final_answer: str


# ─── LLM ─────────────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )


# ─── DataFrame builder ───────────────────────────────────────────────────────

def _build_df(context_data: dict) -> pd.DataFrame:
    rows = []
    for side in ("brand_a", "brand_b"):
        bd = context_data.get(side, {})
        name = context_data.get(f"{side}_name", side)
        sent = bd.get("sentiment", {})
        if hasattr(sent, "positive"):
            pos, neu, neg = sent.positive, sent.neutral, sent.negative
        else:
            pos = sent.get("positive", 0)
            neu = sent.get("neutral", 0)
            neg = sent.get("negative", 0)
        rows.append({
            "brand": name,
            "followers": bd.get("followers", 0),
            "posts_scraped": bd.get("posts_scraped", 0),
            "avg_likes": bd.get("avg_likes", 0),
            "total_engagement": bd.get("total_engagement", 0),
            "positive_sentiment": pos,
            "neutral_sentiment": neu,
            "negative_sentiment": neg,
        })
    return pd.DataFrame(rows)


# ─── Node: classify_intent ───────────────────────────────────────────────────

def classify_intent_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ classify_intent | message: %r", state["message"])
    llm = _llm()
    prompt = f"""You are routing a social media analytics chatbot request. Classify the message below.

Message: "{state['message']}"

Reply with EXACTLY these 5 lines — no other text:
NEEDS_NEWS: yes or no
NEEDS_VIRAL: yes or no
NEEDS_WEB_SEARCH: yes or no
NEEDS_VISUALIZATION: yes or no
TOPIC: the subject to research/create content about, or none
WEB_QUERY: short search query or none

Rules for NEEDS_NEWS = yes (wants latest, credible, real-world information):
- Asks about latest news, recent events, current developments, or facts about a topic
- Asks "what is happening with…", "is it true that…", or for credible/verified info
- Wants up-to-date context before deciding on content
- TOPIC DISCOVERY: asks what topics/trends are popular, emerging, or being talked about now
  (e.g. "what topics are trending", "what are people talking about", "what should we cover",
  "what are we missing", "what else is popular besides what we track")

Rules for NEEDS_VIRAL = yes (wants content ideas to create):
- Asks for content ideas, viral angles, hooks, captions, post ideas, or a campaign
- Asks "what should we post about…", "how do we make this go viral", "give me angles"

Rules for NEEDS_WEB_SEARCH = yes (marketing-strategy research, NOT news):
- Asks about digital campaign best practices, platform algorithms, benchmarks, or tactics used by brands
- TOPIC DISCOVERY: asks what topics/themes/angles competitors or the wider market are covering
  that the team may not be tracking yet

Rules for NEEDS_VISUALIZATION = yes:
- User explicitly requests a chart, graph, plot, bar chart, pie chart, or visual

TOPIC rules:
- If NEEDS_NEWS or NEEDS_VIRAL = yes: extract the core subject (e.g. "vaksin HPV di Indonesia"). Else "none".

WEB_QUERY rules:
- If NEEDS_WEB_SEARCH = yes: a 6–10 word query for marketing intelligence. Else "none".
"""
    result = llm.invoke(prompt)
    content = result.content.strip()

    needs_news = False
    needs_viral = False
    needs_web = False
    needs_viz = False
    topic = ""
    web_query = state["message"]

    for line in content.splitlines():
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()
        if key == "NEEDS_NEWS":
            needs_news = "yes" in val.lower()
        elif key == "NEEDS_VIRAL":
            needs_viral = "yes" in val.lower()
        elif key == "NEEDS_WEB_SEARCH":
            needs_web = "yes" in val.lower()
        elif key == "NEEDS_VISUALIZATION":
            needs_viz = "yes" in val.lower()
        elif key == "TOPIC" and val.lower() != "none" and val:
            topic = val
        elif key == "WEB_QUERY" and val.lower() != "none" and val:
            web_query = val

    # Viral content must be grounded in credible news → force a news pass.
    if needs_viral:
        needs_news = True
    if (needs_news or needs_viral) and not topic:
        topic = state["message"]

    logger.info(
        "[agentic_chat] ✔ classify_intent | news=%s viral=%s web=%s viz=%s topic=%r",
        needs_news, needs_viral, needs_web, needs_viz, topic,
    )
    return {
        **state,
        "needs_news": needs_news,
        "needs_viral": needs_viral,
        "needs_web_search": needs_web,
        "needs_visualization": needs_viz,
        "topic": topic,
        "web_query": web_query,
    }


# ─── Node: web_search ────────────────────────────────────────────────────────

_MARKETING_DOMAINS = [
    "hootsuite.com",
    "sproutsocial.com",
    "buffer.com",
    "socialmediaexaminer.com",
    "hubspot.com",
    "semrush.com",
    "sprinklr.com",
    "later.com",
    "marketingland.com",
    "contentmarketinginstitute.com",
    "searchengineland.com",
    "socialbakers.com",
    "brandwatch.com",
    "mention.com",
    "blog.google",
    "techcrunch.com",
    "forbes.com",
    "businessinsider.com",
]


def web_search_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ web_search | query: %r", state["web_query"])
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("[agentic_chat] ✘ web_search | TAVILY_API_KEY not set")
        return {**state, "web_results": []}
    try:
        os.environ["TAVILY_API_KEY"] = api_key

        # First pass: search within marketing intelligence domains
        tavily_focused = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_domains=_MARKETING_DOMAINS,
        )
        raw = tavily_focused.invoke(state["web_query"])

        # If focused search returns < 3 results, supplement with an open search
        if len(raw) < 3:
            logger.info("[agentic_chat]   web_search | focused returned %d, running open search", len(raw))
            tavily_open = TavilySearchResults(max_results=4, search_depth="advanced")
            raw_open = tavily_open.invoke(state["web_query"])
            seen = {r.get("url") for r in raw}
            raw += [r for r in raw_open if r.get("url") not in seen]

        results = [
            {
                "title": r.get("title") or r.get("url", "Source"),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],
            }
            for r in raw[:6]
        ]
        logger.info("[agentic_chat] ✔ web_search | %d results", len(results))
        return {**state, "web_results": results}
    except Exception as e:
        logger.error("[agentic_chat] ✘ web_search | %s", e)
        return {**state, "web_results": []}


# ─── Node: news — deep credible-news intake ──────────────────────────────────

def news_node(state: AgenticChatState) -> AgenticChatState:
    topic = state.get("topic") or state["message"]
    logger.info("[agentic_chat] ▶ news | topic: %r", topic)
    try:
        brief = ni.gather_news(topic, days=7, max_articles=10)
        logger.info(
            "[agentic_chat] ✔ news | %d articles, %d verified facts",
            len(brief.get("articles", [])), len(brief.get("verified_facts", [])),
        )
        return {**state, "news_brief": brief}
    except Exception as e:  # noqa: BLE001
        logger.error("[agentic_chat] ✘ news | %s", e)
        return {**state, "news_brief": {}}


# ─── Node: viral — angle engine + safety guardrail ───────────────────────────

def _internal_context_line(state: AgenticChatState) -> str:
    """One-line summary of internal scraped data for grounding viral angles."""
    cd = state.get("context_data") or {}
    if not cd.get("brand_a"):
        return ""
    try:
        df = _build_df(cd)
        return "INTERNAL SCRAPED METRICS:\n" + df.to_string(index=False)
    except Exception:  # noqa: BLE001
        return ""


def viral_node(state: AgenticChatState) -> AgenticChatState:
    topic = state.get("topic") or state["message"]
    logger.info("[agentic_chat] ▶ viral | topic: %r", topic)
    try:
        meta = state.get("context_meta", {})
        brand = meta.get("brand_a_name") or "Kalventis (@kenapaharusvaksin)"
        angles = ni.viral_pipeline(
            topic,
            news_brief=state.get("news_brief"),
            internal_context=_internal_context_line(state),
            brand=brand,
            n=4,
        )
        logger.info("[agentic_chat] ✔ viral | %d angles", len(angles))
        return {**state, "viral_angles": angles}
    except Exception as e:  # noqa: BLE001
        logger.error("[agentic_chat] ✘ viral | %s", e)
        return {**state, "viral_angles": []}


# ─── Node: visualize — code-gen + exec ───────────────────────────────────────

def visualize_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ visualize | request: %r", state["message"])

    if not state.get("context_data") or not state["context_data"].get("brand_a"):
        logger.warning("[agentic_chat] ✘ visualize | no context_data")
        return {**state, "viz_b64": None}

    try:
        df = _build_df(state["context_data"])
        logger.info("[agentic_chat]   df | brands=%s", df["brand"].tolist())

        # Step 1: ask DeepSeek to write the chart code
        brands = df["brand"].tolist()
        code_prompt = (
            "You are a Python data visualization expert.\n"
            "Write matplotlib code to fulfill the user's chart request.\n\n"
            f"User request: {state['message']}\n\n"
            "Pre-loaded variables (do NOT redefine them):\n"
            "  plt  — matplotlib.pyplot\n"
            "  pd   — pandas\n"
            f"  df   — pandas DataFrame with integer index (0, 1, ...):\n"
            f"    columns: {list(df.columns)}\n"
            f"{df.to_string(index=True)}\n\n"
            "IMPORTANT — correct data access patterns:\n"
            f"  brands   = df['brand'].tolist()      # → {brands}\n"
            f"  followers = df['followers'].tolist()  # → {df['followers'].tolist()}\n"
            "  Do NOT use df.loc[brand_name, col] — the index is 0, 1, not brand names.\n"
            "  Use df['column'].tolist() or df['column'].values to extract data.\n\n"
            "Styling rules:\n"
            "  - Figure size: fig, ax = plt.subplots(figsize=(7, 4))\n"
            "  - Brand colors: '#2557d6' first brand, '#12a594' second brand\n"
            "  - Remove top and right spines\n"
            "  - Add value labels on bars\n"
            "  - plt.tight_layout() before plt.show()\n"
            "  - End with plt.show()\n\n"
            "Output ONLY raw Python code — no markdown fences, no explanations.\n"
        )
        raw = _llm().invoke(code_prompt).content.strip()
        code = re.sub(r"^```(?:python)?\n?", "", raw, flags=re.MULTILINE)
        code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE).strip()
        logger.info("[agentic_chat]   generated %d chars of code", len(code))
        logger.debug("[agentic_chat]   code:\n%s", code)

        # Step 2: exec the code; intercept plt.show() to capture the PNG
        captured: list = []
        _orig_show = plt.show

        def _capture_show(*args, **kwargs):
            try:
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=72, bbox_inches="tight")
                buf.seek(0)
                captured.append(base64.b64encode(buf.read()).decode())
                logger.info("[agentic_chat] ✔ plt.show() captured")
            except Exception as cap_err:
                logger.warning("[agentic_chat] ✘ capture error: %s", cap_err)
            finally:
                plt.close("all")

        plt.show = _capture_show
        try:
            exec(  # noqa: S102
                compile(code, "<viz_code>", "exec"),
                {"df": df.copy(), "plt": plt, "pd": pd, "matplotlib": matplotlib, "io": io},
            )
            logger.info("[agentic_chat] ✔ exec done | charts: %d", len(captured))
        except Exception as exec_err:
            logger.error("[agentic_chat] ✘ exec failed: %s\ncode:\n%s", exec_err, code)
        finally:
            plt.show = _orig_show
            plt.close("all")

        viz_b64 = captured[0] if captured else None
        if viz_b64:
            logger.info("[agentic_chat] ✔ visualize | chart %d chars b64", len(viz_b64))
        else:
            logger.warning("[agentic_chat] ✘ visualize | no chart (plt.show not called?)")

        return {**state, "viz_b64": viz_b64}

    except Exception as e:
        logger.error("[agentic_chat] ✘ visualize | outer error: %s", e)
        plt.close("all")
        return {**state, "viz_b64": None}


# ─── Node: synthesize ────────────────────────────────────────────────────────

def synthesize_node(state: AgenticChatState) -> AgenticChatState:
    logger.info(
        "[agentic_chat] ▶ synthesize | web=%d chart=%s",
        len(state.get("web_results") or []),
        bool(state.get("viz_b64")),
    )
    llm = _llm()
    meta = state.get("context_meta", {})
    period = meta.get("period", "current period")
    brand_a = meta.get("brand_a_name", "Brand A")
    brand_b = meta.get("brand_b_name", "Brand B")

    def _strip_images(text: str) -> str:
        return re.sub(r'!\[.*?\]\(data:image/[^)]+\)', '[chart]', text).strip()

    history_block = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {_strip_images(m.get('content', ''))}"
        for m in (state.get("history") or [])[-10:]
    )

    web_block = ""
    if state.get("web_results"):
        web_block = "\n\n=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(state["web_results"], 1):
            web_block += f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}\n\n"

    news_block = ni.brief_to_prompt_block(state.get("news_brief") or {})

    viral_block = ""
    angles = state.get("viral_angles") or []
    if angles:
        viral_block = "\n\n=== VIRAL CONTENT ANGLES (already safety-reviewed) ===\n"
        for i, a in enumerate(angles, 1):
            safety = a.get("safety", {})
            viral_block += (
                f"[{i}] ({a.get('virality_score','?')}/100) {a.get('angle','')}\n"
                f"    Platform: {a.get('platform','')} | Format: {a.get('format','')}\n"
                f"    Hook: {a.get('hook','')}\n"
                f"    Safety: {safety.get('verdict','?')}"
                + (f" — {safety.get('fix','')}" if safety.get('fix') else "")
                + "\n"
            )

    has_chart = bool(state.get("viz_b64")) and len(state.get("viz_b64", "")) > 100

    if has_chart:
        chart_instruction = (
            "A real matplotlib chart is already attached below your answer — the user will see it. "
            "IMPORTANT: Do NOT include any chart, graph, ASCII art, text bars, dashes, "
            "or any visual representation in your text. Write only a brief text insight."
        )
    else:
        chart_instruction = (
            "No chart image is available. "
            "Do NOT create ASCII charts, text bars, dashes, or any text-based visual. "
            "State numbers in plain bullet points only."
        )

    has_web = bool(state.get("web_results"))
    has_news = bool(state.get("news_brief", {}).get("articles"))
    has_viral = bool(angles)

    system_persona = (
        "You are a senior digital marketing strategist and social media analyst. "
        "You specialize in Instagram and TikTok brand growth, viral content strategy, "
        "paid and organic campaign design, influencer marketing, and competitive intelligence. "
        "IMPORTANT: This chat is a TOPIC-DISCOVERY tool for a digital marketing team. They already "
        "know the topics and themes in their internal scraped data, so do NOT just restate those. "
        "Your primary job is to surface popular and emerging topics from the wider public conversation "
        "that go BEYOND what they already track — trending subjects, adjacent angles, audience questions, "
        "and competitor topics they may be missing. Lead with these new external topics and industry "
        "research (HootSuite, Sprout Social, SEMrush, HubSpot, Google, news), then use the internal "
        "scraped metrics only as the baseline of 'what they already cover' to contrast against and to "
        "pinpoint the gaps and opportunities."
    )

    news_instruction = (
        "A CREDIBLE NEWS BRIEF is provided. Ground every factual statement in it. "
        "Only state VERIFIED FACTS as fact; clearly hedge anything from UNVERIFIED claims "
        "(e.g. 'reportedly', 'not yet confirmed'). Never invent facts beyond the brief. "
        if has_news else ""
    )
    viral_instruction = (
        "VIRAL CONTENT ANGLES (already safety-reviewed) are provided. Present them as the core of your answer: "
        "lead with the highest virality_score, include the hook verbatim, and respect each Safety verdict — "
        "for 'caution' apply the fix, and do NOT recommend any 'block' angle (explain briefly why instead). "
        if has_viral else ""
    )

    answer_instructions = (
        f"{chart_instruction} {news_instruction}{viral_instruction}"
        "Structure your answer in this order: "
        "(1) Lead with NEW external topics worth their attention — trending subjects, emerging angles, "
        "audience questions, and competitor themes that go BEYOND the topics already in their internal data. "
        "For each, note why it is gaining traction now. "
        "(2) Then map these back to the internal data — call out explicitly which are gaps (popular outside, "
        "absent from their current themes) versus topics they already cover. "
        "Use bullet points. Reference specific external benchmarks or numbers, then contrast with the internal numbers. "
        "If recommending a topic or campaign tactic, explain WHY it is a fresh opportunity this team is not already pursuing."
    )

    full_prompt = (
        f"{system_persona}\n\n"
        f"{state['context_block']}{web_block}{news_block}{viral_block}\n\n"
        f"{history_block}\n\n"
        f"User: {state['message']}\n\n"
        f"{answer_instructions}\n\n"
        f'After your answer add a "---" divider then a "**Sources:**" section:\n'
        f'- Always include: "📊 Internal scraped data · {brand_a} & {brand_b} · {period}"\n'
        + ('- For each web result that contributed: "🌐 [title](url)"\n' if has_web else "")
        + ('- For each news article that contributed, cite it as "🌐 [title](url)"\n' if has_news else "")
        + "\nAssistant:"
    )

    response = llm.invoke(full_prompt)
    answer = response.content.strip()

    b64 = state.get("viz_b64") or ""
    if len(b64) > 100:
        answer += f"\n\n![Visualization](data:image/png;base64,{b64})"

    logger.info("[agentic_chat] ✔ synthesize | %d chars", len(answer))
    return {**state, "final_answer": answer}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _next_after(state: AgenticChatState, *, skip: set[str]) -> str:
    """Pick the next stage, skipping stages already visited."""
    if state.get("needs_news") and "news" not in skip:
        return "news"
    if state.get("needs_viral") and "viral" not in skip:
        return "viral"
    if state.get("needs_web_search") and "web_search" not in skip:
        return "web_search"
    if state.get("needs_visualization") and "visualize" not in skip:
        return "visualize"
    return "synthesize"


def _route_after_classify(state: AgenticChatState) -> str:
    return _next_after(state, skip=set())


def _route_after_news(state: AgenticChatState) -> str:
    return _next_after(state, skip={"news"})


def _route_after_viral(state: AgenticChatState) -> str:
    return _next_after(state, skip={"news", "viral"})


def _route_after_web(state: AgenticChatState) -> str:
    return _next_after(state, skip={"news", "viral", "web_search"})


# ─── Graph ───────────────────────────────────────────────────────────────────

def _compile() -> StateGraph:
    g = StateGraph(AgenticChatState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("news", news_node)
    g.add_node("viral", viral_node)
    g.add_node("web_search", web_search_node)
    g.add_node("visualize", visualize_node)
    g.add_node("synthesize", synthesize_node)

    _stage_map = {
        "news": "news", "viral": "viral", "web_search": "web_search",
        "visualize": "visualize", "synthesize": "synthesize",
    }

    g.set_entry_point("classify_intent")
    g.add_conditional_edges("classify_intent", _route_after_classify, _stage_map)
    g.add_conditional_edges("news", _route_after_news, _stage_map)
    g.add_conditional_edges("viral", _route_after_viral, _stage_map)
    g.add_conditional_edges("web_search", _route_after_web, _stage_map)
    g.add_edge("visualize", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH: Optional[StateGraph] = None


def _get_graph() -> StateGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _compile()
    return _GRAPH


# ─── Public API ──────────────────────────────────────────────────────────────

def run_agentic_chat(
    *,
    message: str,
    history: list,
    context_block: str,
    context_data: dict,
    context_meta: dict,
) -> str:
    logger.info(
        "[agentic_chat] ═══ new request | brands: %s vs %s | history: %d turns",
        context_meta.get("brand_a_name", "?"),
        context_meta.get("brand_b_name", "?"),
        len(history),
    )
    result = _get_graph().invoke({
        "message": message,
        "history": history,
        "context_block": context_block,
        "context_data": context_data,
        "context_meta": context_meta,
        "needs_web_search": False,
        "needs_visualization": False,
        "needs_news": False,
        "needs_viral": False,
        "topic": "",
        "web_query": message,
        "web_results": [],
        "news_brief": {},
        "viral_angles": [],
        "viz_b64": None,
        "final_answer": "",
    })
    logger.info("[agentic_chat] ═══ done")
    return result["final_answer"]
