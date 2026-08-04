# Weekend Planner Agent

ReAct / multi-agent system (LangGraph) that recommends a movie, game, book,
restaurant, or activity in the Eastern Province based on your mood.

**Model:** `gemini-flash-latest`

## Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # add GOOGLE_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
```

## Task 1 — ReAct agent, single node

**Tools:** Tavily (places, movies/games), Open Library (books), Open-Meteo (weather)
**Files:** `weekend_planner.py` (agent logic) · `task1_weekend_planner.ipynb` (demo/deliverable)

Covers: 4 real tools, custom state + reducer, checkpointer (MemorySaver → SqliteSaver),
summarization strategy, recursion limit, intentional tool failure recovery.

Run: `jupyter notebook task1_weekend_planner.ipynb`

## Task 2 — Multi-agent system

**Nodes:** `router` (weather-grounded routing) → `go_out_dispatch` / `stay_home_dispatch`
(parallel fan-out) → `place_agent`, `book_agent`, `media_agent` → `critic` (loop-back,
capped) → `approve_and_remember` (HITL breakpoint + long-term store) → `respond`

**Files:** `weekend_planner_task2.py` (agent logic, imports Task 1's tools) ·
`task2_weekend_planner.ipynb` (demo/deliverable)

Covers: conditional routing (both branches demoed), parallel step, critic with
hard-capped loop-back, HITL breakpoint, long-term memory across sessions, time-travel
debugging of a real failure.

Run: `jupyter notebook task2_weekend_planner.ipynb`

## Studio (visual graph, both tasks)
```bash
langgraph dev --no-reload
```

## Files
- `weekend_planner.py` — Task 1: tools, state, graph
- `task1_weekend_planner.ipynb` — Task 1 deliverable notebook
- `weekend_planner_task2.py` — Task 2: multi-agent graph (imports from weekend_planner.py)
- `task2_weekend_planner.ipynb` — Task 2 deliverable notebook
- `langgraph.json` — Studio config (both graphs)
- `requirements.txt`
