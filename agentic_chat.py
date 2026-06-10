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
    prompt = f"""You are routing an analytics chatbot request. Classify the message below.

Message: "{state['message']}"

Reply with EXACTLY these 3 lines — no other text:
NEEDS_WEB_SEARCH: yes or no
NEEDS_VISUALIZATION: yes or no
WEB_QUERY: short search query or none

Rules:
- NEEDS_WEB_SEARCH = yes → question requires current news, external trends, recent events, or facts outside the internal social media metrics dashboard
- NEEDS_VISUALIZATION = yes → user explicitly requests a chart, graph, plot, or visual
- WEB_QUERY → 5–8 word search query if web search needed, otherwise "none"
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
        "[agentic_chat] ✔ classify_intent | web=%s viz=%s query=%r",
        needs_web, needs_viz, web_query,
    )
    return {**state, "needs_web_search": needs_web, "needs_visualization": needs_viz, "web_query": web_query}


# ─── Node: web_search ────────────────────────────────────────────────────────

def web_search_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ web_search | query: %r", state["web_query"])
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("[agentic_chat] ✘ web_search | TAVILY_API_KEY not set")
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
        logger.info("[agentic_chat] ✔ web_search | %d results", len(results))
        return {**state, "web_results": results}
    except Exception as e:
        logger.error("[agentic_chat] ✘ web_search | %s", e)
        return {**state, "web_results": []}


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

    full_prompt = (
        f"{state['context_block']}{web_block}\n\n"
        f"{history_block}\n\n"
        f"User: {state['message']}\n\n"
        f"{chart_instruction} "
        f"Answer directly and practically. Use bullet points where helpful. "
        f"Reference specific numbers when relevant.\n\n"
        f'After your answer add a "---" divider then a "**Sources:**" section:\n'
        f'- Always include: "📊 Internal scraped data · {brand_a} & {brand_b} · {period}"\n'
        '- For each web result that contributed: "🌐 [title](url)"\n\n'
        "Assistant:"
    )

    response = llm.invoke(full_prompt)
    answer = response.content.strip()

    b64 = state.get("viz_b64") or ""
    if len(b64) > 100:
        answer += f"\n\n![Visualization](data:image/png;base64,{b64})"

    logger.info("[agentic_chat] ✔ synthesize | %d chars", len(answer))
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
        "web_query": message,
        "web_results": [],
        "viz_b64": None,
        "final_answer": "",
    })
    logger.info("[agentic_chat] ═══ done")
    return result["final_answer"]
