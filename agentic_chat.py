"""Agentic chat for Social Listening — LangGraph + Tavily web search + pandas visualization."""

import base64
import concurrent.futures
import io
import logging
import os
from typing import Optional, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

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
    web_query: str
    web_results: list
    viz_b64: Optional[str]
    final_answer: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY", ""),
    )


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
    prompt = f"""You are routing an analytics chatbot request. Classify the message below.

Message: "{state['message']}"

Reply with EXACTLY these 3 lines — no other text:
NEEDS_WEB_SEARCH: yes or no
NEEDS_VISUALIZATION: yes or no
WEB_QUERY: short search query or none

Rules:
- NEEDS_WEB_SEARCH = yes → question requires current news, external trends, recent events, or facts outside the internal social media metrics dashboard
- NEEDS_VISUALIZATION = yes → user explicitly requests a chart, graph, plot, or visual
- WEB_QUERY → 5–8 word search query if web needed, otherwise "none"
"""
    result = llm.invoke(prompt)
    content = result.content.strip()

    needs_web = False
    needs_viz = False
    web_query = state["message"]

    for line in content.splitlines():
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()
        if key == "NEEDS_WEB_SEARCH":
            needs_web = "yes" in val.lower()
        elif key == "NEEDS_VISUALIZATION":
            needs_viz = "yes" in val.lower()
        elif key == "WEB_QUERY" and val.lower() != "none" and val:
            web_query = val

    logger.info(
        "[agentic_chat] ✔ classify_intent | web_search=%s  visualization=%s  query=%r",
        needs_web, needs_viz, web_query,
    )
    return {**state, "needs_web_search": needs_web, "needs_visualization": needs_viz, "web_query": web_query}


# ─── Node: web_search ────────────────────────────────────────────────────────

def web_search_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ web_search | query: %r", state["web_query"])
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("[agentic_chat] ✘ web_search | TAVILY_API_KEY not set — skipping")
        return {**state, "web_results": []}
    try:
        os.environ["TAVILY_API_KEY"] = api_key
        tavily = TavilySearchResults(k=4)
        raw = tavily.invoke(state["web_query"])
        results = [
            {
                "title": r.get("title") or r.get("url", "Source"),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:600],
            }
            for r in raw[:4]
        ]
        logger.info("[agentic_chat] ✔ web_search | got %d results", len(results))
        return {**state, "web_results": results}
    except Exception as e:
        logger.error("[agentic_chat] ✘ web_search | error: %s", e)
        return {**state, "web_results": []}


# ─── Node: visualize ─────────────────────────────────────────────────────────

def _extract_code(raw: str) -> Optional[str]:
    start = raw.find("```python")
    if start == -1:
        return None
    start += len("```python")
    end = raw.find("```", start)
    if end == -1:
        return None
    return raw[start:end].strip()


def _exec_chart(code: str, df: pd.DataFrame) -> str:
    """exec() the code and return base64 PNG. Pre-injects fig/ax so Gemini never hits NameError."""
    # Pre-create fig and ax — Gemini can use them without defining, or redefine if needed
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fc")
    env = {"pd": pd, "df": df, "plt": plt, "io": io, "base64": base64, "fig": fig, "ax": ax}
    exec(code, {}, env)  # nosec — controlled internal use only
    # Use result if it's a Figure, otherwise fall back to the pre-created fig
    out_fig = env.get("result")
    if not isinstance(out_fig, plt.Figure):
        out_fig = fig
    buf = io.BytesIO()
    out_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    viz_b64 = base64.b64encode(buf.read()).decode()
    plt.close("all")
    return viz_b64


def _generate_chart(llm, df: pd.DataFrame, message: str) -> Optional[str]:
    """Ask Gemini to generate matplotlib code, exec() it."""
    prompt = f"""You are a Python data visualization expert.

DataFrame `df` is already loaded with social media brand metrics:
Columns: {list(df.columns)}
Data:
{df.to_string(index=False)}

Write Python matplotlib code for: "{message}"

These variables are PRE-CREATED — use them directly, do NOT redefine with plt.subplots():
  fig  → matplotlib Figure (figsize 8×5, white background)
  ax   → matplotlib Axes (light grey background)
  df   → the DataFrame above
  plt  → matplotlib.pyplot (Agg backend, already imported)

STYLING:
- Brand colors: '#2557d6' for first brand, '#12a594' for second
- bars/wedges alpha=0.88
- ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
- Add value labels on bars/slices
- ax.set_title(..., fontsize=14, fontweight='bold', color='#111827')
- Do NOT call plt.show() or plt.subplots()
- Set result = fig at the end

Return ONLY a ```python ... ``` code block, no explanations."""

    response = llm.invoke(prompt)
    code = _extract_code(response.content.strip())
    if not code:
        logger.warning("[agentic_chat] ✘ visualize | no code block in LLM response")
        return None

    logger.info("[agentic_chat]   visualize | executing code (%d chars)", len(code))
    try:
        return _exec_chart(code, df)
    except Exception as e:
        logger.warning("[agentic_chat] ✘ visualize | exec failed: %s", e)
        plt.close("all")
        return None


def visualize_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ visualize | request: %r", state["message"])
    if not state.get("context_data") or not state["context_data"].get("brand_a"):
        logger.warning("[agentic_chat] ✘ visualize | no context_data available")
        return {**state, "viz_b64": None}

    try:
        llm = _llm()
        df = _build_df(state["context_data"])
        logger.info("[agentic_chat]   visualize | DataFrame shape: %s, brands: %s", df.shape, df["brand"].tolist())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate_chart, llm, df, state["message"])
            try:
                viz_b64 = future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.warning("[agentic_chat] ✘ visualize | timed out after 30s — skipping chart")
                viz_b64 = None

        if viz_b64 and len(viz_b64) > 100:
            logger.info("[agentic_chat] ✔ visualize | chart captured (%d chars b64)", len(viz_b64))
        else:
            logger.warning("[agentic_chat] ✘ visualize | no chart captured")
            viz_b64 = None
        return {**state, "viz_b64": viz_b64}

    except Exception as e:
        logger.error("[agentic_chat] ✘ visualize | error: %s", e)
        return {**state, "viz_b64": None}


# ─── Node: synthesize ────────────────────────────────────────────────────────

def synthesize_node(state: AgenticChatState) -> AgenticChatState:
    logger.info(
        "[agentic_chat] ▶ synthesize | web_results=%d  has_chart=%s",
        len(state.get("web_results") or []),
        bool(state.get("viz_b64")),
    )
    llm = _llm()
    meta = state.get("context_meta", {})
    period = meta.get("period", "current period")
    brand_a = meta.get("brand_a_name", "Brand A")
    brand_b = meta.get("brand_b_name", "Brand B")

    import re as _re
    def _strip_images(text: str) -> str:
        return _re.sub(r'!\[.*?\]\(data:image/[^)]+\)', '[chart]', text).strip()

    history_block = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {_strip_images(m.get('content', ''))}"
        for m in (state.get("history") or [])[-10:]
    )

    web_block = ""
    if state.get("web_results"):
        web_block = "\n\n=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(state["web_results"], 1):
            web_block += f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}\n\n"

    full_prompt = (
        f"{state['context_block']}{web_block}\n\n"
        f"{history_block}\n\n"
        f"User: {state['message']}\n\n"
        "Answer directly and practically. Use bullet points where helpful. "
        "Reference specific numbers when relevant.\n\n"
        f'After your answer add a "---" divider then a "**Sources:**" section:\n'
        f"- Always include: \"📊 Internal scraped data · {brand_a} & {brand_b} · {period}\"\n"
        "- For each web result that actually contributed to your answer add: "
        '"🌐 [title](url)"\n\n'
        "Assistant:"
    )

    response = llm.invoke(full_prompt)
    answer = response.content.strip()

    b64 = state.get("viz_b64") or ""
    if len(b64) > 100:
        answer += f"\n\n![Visualization](data:image/png;base64,{b64})"

    logger.info("[agentic_chat] ✔ synthesize | answer length: %d chars", len(answer))
    return {**state, "final_answer": answer}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_after_classify(state: AgenticChatState) -> str:
    if state["needs_web_search"]:
        return "web_search"
    if state["needs_visualization"]:
        return "visualize"
    return "synthesize"


def _route_after_web(state: AgenticChatState) -> str:
    return "visualize" if state["needs_visualization"] else "synthesize"


# ─── Graph ───────────────────────────────────────────────────────────────────

def _compile() -> StateGraph:
    g = StateGraph(AgenticChatState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("web_search", web_search_node)
    g.add_node("visualize", visualize_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"web_search": "web_search", "visualize": "visualize", "synthesize": "synthesize"},
    )
    g.add_conditional_edges(
        "web_search",
        _route_after_web,
        {"visualize": "visualize", "synthesize": "synthesize"},
    )
    g.add_edge("visualize", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_GRAPH = None


def _get_graph():
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
    graph = _get_graph()
    result = graph.invoke({
        "message": message,
        "history": history,
        "context_block": context_block,
        "context_data": context_data,
        "context_meta": context_meta,
        "needs_web_search": False,
        "needs_visualization": False,
        "web_query": message,
        "web_results": [],
        "viz_b64": None,
        "final_answer": "",
    })
    logger.info("[agentic_chat] ═══ done")
    return result["final_answer"]
