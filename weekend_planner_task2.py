"""
Weekend Planner Agent — Task 2 (Human-in-the-Loop, Parallelism & Multi-Agent)

Extends Task 1 by importing its tools and reducer (weekend_planner.py is left
untouched). This is a genuine multi-agent system, not a single ReAct loop:

  router              -> decides "go_out" / "go_out_media" / "stay_home" using
                          a REAL weather call + keywords from the user's
                          message. If ambiguous, asks the user instead of
                          guessing (different job/tool than the specialists).
  clarify              -> asks the user to pick a branch when the message was
                          ambiguous, then ends the turn and waits for a reply.
  place_agent          -> tool: search_local_spot            (go_out branch)
  go_out_dispatch       -> pass-through hub: fans out to place_agent + media_agent
                          in parallel (go_out_media branch — wants a movie/game
                          out of the house, e.g. cinema or an arcade)
  stay_home_dispatch    -> pass-through hub: fans out to book_agent + media_agent
                          in parallel (stay_home branch)
  book_agent            -> tool: get_book_recommendation
  media_agent           -> tools: get_trending_movie_or_game + log_suggestion
  critic                -> no tool; validates recommendations are non-empty and
                          grounded, can send work back to a specialist (capped)
  approve_and_remember  -> HITL breakpoint (interrupt()) + writes the approved
                          suggestion into a long-term store (crosses threads)
  respond               -> LLM synthesis of the final bilingual answer, no tools

Graph shape:
  START -> router -> clarify (ambiguous) -> END (waits for the user's reply)
                   -> place_agent                              (go_out)
                   -> go_out_dispatch -> [place_agent, media_agent]   (go_out_media, parallel)
                   -> stay_home_dispatch -> [book_agent, media_agent] (stay_home, parallel)
  {place_agent | book_agent | media_agent} -> critic
  critic -> {retry a specialist} | approve_and_remember
  approve_and_remember -> respond -> END
"""

import os
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

# Reuse Task 1's tools and reducer as-is.
from weekend_planner import (
    get_book_recommendation,
    get_weather,
    llm,
    merge_recommendations,
    search_local_spot,
)

try:
    from tavily import TavilyClient
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install tavily-python") from e

_tavily_client = None
_tavily_client_key = None


def _get_tavily_client() -> "TavilyClient":
    global _tavily_client, _tavily_client_key
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment")
    if _tavily_client is None or api_key != _tavily_client_key:
        _tavily_client = TavilyClient(api_key=api_key)
        _tavily_client_key = api_key
    return _tavily_client


from langchain_core.tools import tool


@tool
def get_trending_movie_or_game(media_type: str, preferred_genre: str = "") -> dict:
    """Search for what movie or video game is currently trending, optionally
    filtered by genre.

    Args:
        media_type: either 'movie' or 'game'.
        preferred_genre: optional genre filter, e.g. 'action'. Empty string if none.
    """
    if media_type not in ("movie", "game"):
        raise ValueError("media_type must be 'movie' or 'game'")
    genre_part = f" {preferred_genre}" if preferred_genre else ""
    query = f"أكثر{genre_part} {'أفلام' if media_type == 'movie' else 'ألعاب'} رواجاً الآن في السعودية"
    result = _get_tavily_client().search(query=query, max_results=3, timeout=15)
    return {f"trending_{media_type}": result.get("results", [])}


# ---------------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------------
class Task2State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    mood: str
    city: str
    language: str
    branch: Optional[str]  # "go_out" | "stay_home" | "unclear"
    recommendations: Annotated[dict, merge_recommendations]
    revision_count: int
    critic_feedback: Optional[str]


# ---------------------------------------------------------------------------
# 2. Router — different job (routing) + different tool (weather) than the
#    specialists below. Uses a REAL Open-Meteo call to ground the decision.
#    Gender-inclusive keyword lists (masculine and feminine forms).
#    If the message is ambiguous, we ASK instead of silently guessing from
#    weather alone — weather is shown as context in the clarifying question.
# ---------------------------------------------------------------------------
GO_OUT_WORDS = [
    "اطلع", "أطلع", "طالع", "طالعة", "برا", "مطعم", "نشاط",
    "اخرج", "أخرج", "خارج", "سينما",
    "صالة ألعاب", "أركيد", "مكان ألعاب",
]
STAY_HOME_WORDS = [
    "بيت", "البيت", "اقعد", "أقعد", "قاعد", "قاعدة", "كتاب", "فيلم", "لعبة",
]
BOTH_WORDS = ["الكل", "كل شي", "الاثنين", "كلهم", "برا وداخل"]
NEGATION_WORDS = ["لا ", "ما ", "مو ", "مب "]


def _first_match_index(text, words):
    positions = [text.find(w) for w in words if w in text]
    positions = [p for p in positions if p != -1]
    return min(positions) if positions else None


def _is_negated(text, keyword_index):
    # Looks for a negation word in the 15 characters right before the match —
    # catches phrases like "لا ودي أطلع" (negating "أطلع") without needing
    # real NLP, just a nearby-word heuristic.
    window = text[max(0, keyword_index - 15):keyword_index]
    return any(neg in window for neg in NEGATION_WORDS)

CITY_ALIASES = {
    "Dammam": ["دمام", "الدمام", "dammam"],
    "Khobar": ["خبر", "الخبر", "khobar"],
    "Dhahran": ["ظهران", "الظهران", "dhahran"],
    "Qatif": ["قطيف", "القطيف", "qatif"],
    "Jubail": ["جبيل", "الجبيل", "jubail"],
    "Saihat": ["سيهات", "saihat"],
}


def _extract_city(text: str, fallback: str) -> str:
    text_lower = text.lower()
    for canonical, aliases in CITY_ALIASES.items():
        if any(alias in text or alias in text_lower for alias in aliases):
            return canonical
    return fallback

def router(state: Task2State):
    last_text = state["messages"][-1].content if state["messages"] else ""
    city = _extract_city(last_text, state.get("city", "Dammam"))
    weather_result = get_weather.invoke({"city": city})

    out_idx = _first_match_index(last_text, GO_OUT_WORDS)
    home_idx = _first_match_index(last_text, STAY_HOME_WORDS)

    wants_out = out_idx is not None and not _is_negated(last_text, out_idx)
    wants_home = home_idx is not None and not _is_negated(last_text, home_idx)

    if wants_out and not wants_home:
        branch = "go_out"
    elif wants_home and not wants_out:
     branch = "stay_home"
    elif wants_out and wants_home:
        branch = "go_out" if out_idx > home_idx else "stay_home"
    else:
        branch = "unclear"

    return {
        "branch": branch,
        "city": city,
        "recommendations": weather_result,
        "revision_count": 0,
    }


def route_from_router(state: Task2State):
    if state["branch"] == "unclear":
        return "clarify"
    if state["branch"] == "both":
        return "both_dispatch"
    if state["branch"] == "go_out":
        return "go_out_dispatch"
    return "stay_home_dispatch"


def clarify(state: Task2State):
    weather = state.get("recommendations", {}).get("weather", {})
    temperature = weather.get("temperature")

    if temperature is not None and temperature >= 35:
        suggestion = (
            f"الجو حار الحين ({temperature}°م)، فالأنسب يكون نشاط داخلي — "
            "كتاب أو فيلم بالبيت. يناسب هذا الاقتراح، أو الأفضلية للخروج رغم الحر؟"
        )
    elif temperature is not None:
        suggestion = (
            f"الجو معتدل الحين ({temperature}°م)، فرصة زينة للخروج — "
            "مطعم أو نشاط، أو فيلم بالسينما. يناسب هذا، أو الأفضلية للبقاء بالبيت؟"
        )
    else:
        suggestion = (
            "مو واضح المطلوب اليوم. الخيارات: الخروج لمطعم أو نشاط، الخروج "
            "لمكان فيه فيلم أو ألعاب، أو البقاء بالبيت. أي خيار يناسب؟"
        )

    return {"messages": [AIMessage(content=suggestion)]}


# ---------------------------------------------------------------------------
# 3. Specialists — each a different job + different tool
# ---------------------------------------------------------------------------
def place_agent(state: Task2State):
    try:
        result = search_local_spot.invoke(
            {"activity_type": "مطعم أو نشاط", "city": state.get("city", "Dammam")}
        )
    except Exception as e:
        result = {"place_agent_error": str(e)}
    return {"recommendations": result}


def go_out_dispatch(state: Task2State):
    # Pass-through hub: its only job is to have two outgoing edges, which is
    # what makes place_agent and media_agent run in parallel.
    return {}


def stay_home_dispatch(state: Task2State):
    # Same idea as go_out_dispatch, for the stay-home branch.
    return {}

def both_dispatch(state: Task2State):
    # Same idea, but fans out to all three specialists at once.
    return {}


def book_agent(state: Task2State):
    try:
        result = get_book_recommendation.invoke({"topic": "fiction"})
    except Exception as e:
        result = {"book_agent_error": str(e)}
    return {"recommendations": result}


def media_agent(state: Task2State):
    try:
        mood = state.get("mood", "")
        result = get_trending_movie_or_game.invoke(
            {"media_type": "movie", "preferred_genre": mood}
        )
        titles = [r.get("title", "") for r in result.get("trending_movie", [])[:1]]
        log = {"suggestion_log": titles} if titles else {}
        result = {**result, **log}
    except Exception as e:
        result = {"media_agent_error": str(e)}
    return {"recommendations": result}


# ---------------------------------------------------------------------------
# 4. Critic — no tool; validates the OTHER nodes' work and can loop back
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "go_out": ["spot_suggestion", "trending_movie"],
    "stay_home": ["book_suggestion", "trending_movie"],
    "both": ["spot_suggestion", "book_suggestion", "trending_movie"],
}


def critic(state: Task2State):
    branch = state["branch"]
    rec = state.get("recommendations", {})
    missing = [k for k in REQUIRED_KEYS[branch] if not rec.get(k)]

    if missing and state.get("revision_count", 0) < 2:
        return {
            "revision_count": state.get("revision_count", 0) + 1,
            "critic_feedback": f"ناقص: {', '.join(missing)} — إعادة محاولة",
        }
    return {
        "critic_feedback": "موافق عليه"
        if not missing
        else "تم القبول رغم نقص جزئي (وصلنا الحد الأقصى للمحاولات)"
    }


def route_from_critic(state: Task2State):
    branch = state["branch"]
    rec = state.get("recommendations", {})
    missing = [k for k in REQUIRED_KEYS[branch] if not rec.get(k)]

    if missing and state.get("revision_count", 0) < 2:
        if branch == "go_out":
            return "go_out_dispatch"
        if branch == "both":
            return "both_dispatch"
        return "stay_home_dispatch"
    return "approve_and_remember"


# ---------------------------------------------------------------------------
# 5. HITL breakpoint + long-term memory write
#    `store` is auto-injected by LangGraph because the graph is compiled
#    with store=... and this node declares a `store: BaseStore` parameter.
# ---------------------------------------------------------------------------
def approve_and_remember(state: Task2State, store: BaseStore):
    preview = state.get("recommendations", {})
    decision = interrupt(
        {
            "question": "هل انت موافق على هذي التوصية قبل حفظها بالذاكرة الدائمة؟",
            "preview": preview,
        }
    )
    namespace = ("najaf", "suggestions")
    if decision and str(decision).strip() not in ("رفض", "reject", "no"):
        store.put(namespace, str(hash(str(preview))), {"recommendations": preview})
    return {}


# ---------------------------------------------------------------------------
# 6. Final synthesis — LLM, no tools
# ---------------------------------------------------------------------------
RESPOND_PROMPT_AR = "لخّص التوصيات التالية للمستخدم بجملتين إلى ثلاث، بالعربية فقط: {recommendations}"
RESPOND_PROMPT_EN = "Summarize the following recommendations for the user in two to three sentences, in English only: {recommendations}"


def respond(state: Task2State):
    language = state.get("language", "ar")
    template = RESPOND_PROMPT_AR if language == "ar" else RESPOND_PROMPT_EN
    prompt = template.format(recommendations=state.get("recommendations", {}))
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 7. Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(Task2State)
builder.add_node("router", router)
builder.add_node("clarify", clarify)
builder.add_node("place_agent", place_agent)
builder.add_node("go_out_dispatch", go_out_dispatch)
builder.add_node("stay_home_dispatch", stay_home_dispatch)
builder.add_node("both_dispatch", both_dispatch)
builder.add_node("book_agent", book_agent)
builder.add_node("media_agent", media_agent)
builder.add_node("critic", critic)
builder.add_node("approve_and_remember", approve_and_remember)
builder.add_node("respond", respond)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_from_router,
    ["clarify", "go_out_dispatch", "stay_home_dispatch", "both_dispatch"],
)
builder.add_edge("clarify", END)  # ends the turn; user's next message re-enters at router

# Parallel steps: two plain edges from the same source = both run at once.
builder.add_edge("go_out_dispatch", "place_agent")
builder.add_edge("go_out_dispatch", "media_agent")
builder.add_edge("stay_home_dispatch", "book_agent")
builder.add_edge("stay_home_dispatch", "media_agent")
builder.add_edge("both_dispatch", "place_agent")
builder.add_edge("both_dispatch", "book_agent")
builder.add_edge("both_dispatch", "media_agent")

# Fan-in: all specialist paths converge on critic.
builder.add_edge("place_agent", "critic")
builder.add_edge("book_agent", "critic")
builder.add_edge("media_agent", "critic")


builder.add_conditional_edges(
    "critic",
    route_from_critic,
    ["go_out_dispatch", "stay_home_dispatch", "both_dispatch", "approve_and_remember"],
)
builder.add_edge("approve_and_remember", "respond")
builder.add_edge("respond", END)

# NOTE: compiled WITHOUT a checkpointer or store here — LangGraph API/Studio
# rejects a custom store passed at compile time (it manages persistence
# itself and raises a ValueError otherwise). This means `approve_and_remember`
# writing to a store only works when this graph is compiled explicitly WITH
# a store — which is exactly what the demo notebook does
# (`builder.compile(checkpointer=memory, store=long_term_store)`), since it
# imports `builder` from this module directly instead of using `graph` below.
graph = builder.compile()