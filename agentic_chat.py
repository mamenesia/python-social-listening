"""Agentic chat for Social Listening — LangGraph + Tavily web search."""

import logging
import os
from typing import TypedDict

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
    web_query: str
    web_results: list
    final_answer: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY", ""),
    )


# ─── Node: classify_intent ───────────────────────────────────────────────────

def classify_intent_node(state: AgenticChatState) -> AgenticChatState:
    logger.info("[agentic_chat] ▶ classify_intent | message: %r", state["message"])
    llm = _llm()
    prompt = f"""You are routing an analytics chatbot request. Classify the message below.

Message: "{state['message']}"

Reply with EXACTLY these 2 lines — no other text:
NEEDS_WEB_SEARCH: yes or no
WEB_QUERY: short search query or none

Rules:
- NEEDS_WEB_SEARCH = yes → question requires current news, external trends, recent events, or facts outside the internal social media metrics dashboard
- WEB_QUERY → 5–8 word search query if web needed, otherwise "none"
"""
    result = llm.invoke(prompt)
    content = result.content.strip()

    needs_web = False
    web_query = state["message"]

    for line in content.splitlines():
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()
        if key == "NEEDS_WEB_SEARCH":
            needs_web = "yes" in val.lower()
        elif key == "WEB_QUERY" and val.lower() != "none" and val:
            web_query = val

    logger.info(
        "[agentic_chat] ✔ classify_intent | web_search=%s  query=%r",
        needs_web, web_query,
    )
    return {**state, "needs_web_search": needs_web, "web_query": web_query}


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


# ─── Node: synthesize ────────────────────────────────────────────────────────

def synthesize_node(state: AgenticChatState) -> AgenticChatState:
    logger.info(
        "[agentic_chat] ▶ synthesize | web_results=%d",
        len(state.get("web_results") or []),
    )
    llm = _llm()
    meta = state.get("context_meta", {})
    period = meta.get("period", "current period")
    brand_a = meta.get("brand_a_name", "Brand A")
    brand_b = meta.get("brand_b_name", "Brand B")

    history_block = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
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

    logger.info("[agentic_chat] ✔ synthesize | answer length: %d chars", len(answer))
    return {**state, "final_answer": answer}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_after_classify(state: AgenticChatState) -> str:
    return "web_search" if state["needs_web_search"] else "synthesize"


# ─── Graph ───────────────────────────────────────────────────────────────────

def _compile() -> StateGraph:
    g = StateGraph(AgenticChatState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("web_search", web_search_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"web_search": "web_search", "synthesize": "synthesize"},
    )
    g.add_edge("web_search", "synthesize")
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
        "web_query": message,
        "web_results": [],
        "final_answer": "",
    })
    logger.info("[agentic_chat] ═══ done")
    return result["final_answer"]
