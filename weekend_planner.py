"""
Weekend Planner Agent — Task 1 (Foundations, ReAct Agents & State)

A ReAct agent that recommends a movie, game, book, restaurant, or activity
in Saudi Arabia's Eastern Province based on the user's mood.

This module is imported in two places:
  1. LangGraph Studio, via langgraph.json -> "weekend_planner": "./weekend_planner.py:graph"
  2. The deliverable notebook (task1_weekend_planner.ipynb), which reuses
     `tools`, `llm`, `PlannerState`, `assistant`, and `builder` to demo the
     checkpointer swap, recursion limit, and the intentionally-broken tool.

No API keys are hardcoded here — everything is read from environment
variables, which LangGraph Studio and python-dotenv both load from .env.
"""

import os
from typing import Annotated, Optional, TypedDict

import requests
from langchain_core.messages import AnyMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from tavily import TavilyClient
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install tavily-python") from e


# ---------------------------------------------------------------------------
# 1. Custom state schema + reducer
# ---------------------------------------------------------------------------
def merge_recommendations(existing: Optional[dict], new: Optional[dict]) -> dict:
    """Reducer: merges tool outputs into the shared recommendations dict
    instead of overwriting it. Existing keys survive unless the new dict
    explicitly updates the same key. This matters because more than one
    tool can write to this field across the run.
    """
    existing = existing or {}
    new = new or {}
    merged = dict(existing)
    merged.update(new)
    return merged


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    mood: str
    city: str
    recommendations: Annotated[dict, merge_recommendations]
    summary: str


# ---------------------------------------------------------------------------
# 2. Tools — 4 real external calls, each with a docstring the model reads
# ---------------------------------------------------------------------------
_tavily_client = None
_tavily_client_key = None


def _get_tavily_client() -> "TavilyClient":
    global _tavily_client, _tavily_client_key
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment")
    # Rebuild the client if it's the first call OR the key changed since
    # the last time we built it (e.g. after a deliberate break-and-restore
    # test) — otherwise a stale cached client silently keeps using the old key.
    if _tavily_client is None or api_key != _tavily_client_key:
        _tavily_client = TavilyClient(api_key=api_key)
        _tavily_client_key = api_key
    return _tavily_client


@tool
def search_local_spot(activity_type: str, city: str = "Dammam") -> dict:
    """Search for a real restaurant or activity spot in the Eastern Province
    of Saudi Arabia (Dammam, Khobar, Dhahran, Qatif, Jubail...).

    Args:
        activity_type: what kind of place, e.g. 'quiet cafe', 'outdoor activity', 'family restaurant'.
        city: which Eastern Province city to search in. Defaults to Dammam.
    """
    query = f"{activity_type} في {city}، المنطقة الشرقية السعودية"
    result = _get_tavily_client().search(query=query, max_results=3, timeout=15)
    return {"spot_suggestion": result.get("results", [])}


@tool
def search_movie_or_game(mood: str, media_type: str) -> dict:
    """Search the web for a movie or video game recommendation that fits a mood.

    Args:
        mood: the person's current mood, e.g. 'bored', 'tired', 'want an adrenaline rush'.
        media_type: either 'movie' or 'game'.
    """
    if media_type not in ("movie", "game"):
        raise ValueError("media_type must be 'movie' or 'game'")
    query = f"best {media_type} to watch or play when you feel {mood} 2026"
    result = _get_tavily_client().search(query=query, max_results=3)
    return {f"{media_type}_suggestion": result.get("results", [])}


@tool
def get_book_recommendation(topic: str) -> dict:
    """Get a real book recommendation from the Open Library API based on a topic or genre.

    Args:
        topic: subject or genre to search for, e.g. 'mystery', 'self help', 'arabic poetry'.
    """
    resp = requests.get(
        "https://openlibrary.org/subjects/" + topic.lower().replace(" ", "_") + ".json",
        params={"limit": 3},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    works = [
        {"title": w.get("title"), "authors": [a.get("name") for a in w.get("authors", [])]}
        for w in data.get("works", [])[:3]
    ]
    return {"book_suggestion": works}


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for an Eastern Province city using Open-Meteo,
    to help decide between an indoor or outdoor activity.

    Args:
        city: city name, e.g. 'Dammam', 'Khobar', 'Qatif'.
    """
    coords = {
        "dammam": (26.4207, 49.9777),
        "khobar": (26.2172, 50.1971),
        "dhahran": (26.2361, 50.0393),
        "qatif": (26.5205, 49.9865),
        "jubail": (27.0046, 49.6600),
        "saihat": (26.5372, 50.0208),
    }
    lat, lon = coords.get(city.lower(), coords["dammam"])
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json().get("current_weather", {})
    return {"weather": current}


tools = [search_local_spot, search_movie_or_game, get_book_recommendation, get_weather]


# ---------------------------------------------------------------------------
# 3. Model + message trimming
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.4)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "أنت مساعد تخطيط عطلة نهاية الأسبوع لسكان المنطقة الشرقية بالسعودية. "
    "يخبرك المستخدم بمزاجه (مثلاً 'طفشانة')، وعليك استخدام أدواتك لتأسيس كل "
    "توصية على نتيجة بحث أو استدعاء API حقيقي — لا تخترع مطعماً أو فيلماً أو "
    "لعبة أو حالة طقس أبداً. استخدم أداة الطقس لتقرر بين نشاط داخلي أو "
    "خارجي. رد بتوصية مختصرة بالعربية والإنجليزية."
)


def assistant(state: PlannerState):
    summary = state.get("summary", "")
    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nملخص المحادثة حتى الآن: {summary}"
    response = llm_with_tools.invoke([SystemMessage(content=system_content)] + state["messages"])
    return {"messages": [response]}

def summarize_conversation(state: PlannerState):
    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"هذا ملخص للمحادثة السابقة: {summary}\n\n"
            "وسّعي الملخص أعلاه ليشمل الرسائل الجديدة أعلاه، بجملتين كحد أقصى:"
        )
    else:
        summary_prompt = "لخصي المحادثة أعلاه بجملتين كحد أقصى:"

    messages = state["messages"] + [SystemMessage(content=summary_prompt)]
    response = llm.invoke(messages)  


    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def route_after_assistant(state: PlannerState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    if len(state["messages"]) > 6:
        return "summarize_conversation"
    return END


# ---------------------------------------------------------------------------
# 4. Build the graph
#    NOTE: compiled WITHOUT an explicit checkpointer here on purpose —
#    LangGraph Studio manages its own persistence layer for graphs loaded
#    this way. The notebook demonstrates MemorySaver -> SqliteSaver
#    explicitly using `builder` imported from this module.
# ---------------------------------------------------------------------------
builder = StateGraph(PlannerState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant", route_after_assistant, ["tools", "summarize_conversation", END]
)
builder.add_edge("tools", "assistant")
builder.add_edge("summarize_conversation", END)

graph = builder.compile()
