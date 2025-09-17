import sys
from copy import copy
from queue import Queue
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal, Callable, Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from textwrap import dedent
import json
from datetime import datetime
import re
from jinja2 import Environment, FileSystemLoader
import os
import argparse

# Rich
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"

# ===== Helpers to load config =====

def load_roles(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Basic validation
    if "game_master" not in cfg or "historian" not in cfg or "players" not in cfg:
        raise ValueError("JSON must contain 'game_master', 'players', and 'historian'.")
    if not isinstance(cfg["players"], list) or len(cfg["players"]) == 0:
        raise ValueError("'players' must be a non-empty list.")
    # Normalize names to simple tokens for internal keys
    def norm(n: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", n.strip()).lower()
    cfg["_gm_key"] = norm(cfg["game_master"]["name"])
    cfg["_hist_key"] = norm(cfg["historian"]["name"])
    cfg["_player_keys"] = [norm(p["name"]) for p in cfg["players"]]
    return cfg

# ====== State schema (generic) ======

@dataclass
class AgentState:
    # dynamic role → system prompt map
    systems: Dict[str, str]
    # ordered list of player keys (turn order)
    player_keys: List[str]
    gm_key: str
    hist_key: str

    llm_model: str
    max_iterations: int
    user_id: str

    # messages per role key
    msgs: Dict[str, List[str]] = field(default_factory=dict)

    approximate_cost: float = 0.0
    chat_history_log_file: str = ""
    current_iteration: int = 1
    historical_account: str = ""
    latex_filename: str = ""
    print: Callable[[Any], None] = print  # default to standard print

    console: Console | None = None
    role_styles: dict = field(default_factory=dict)   # role_key -> style name
    kind_styles: dict = field(default_factory=dict)   # "header"/"msg"/"cost"/"debug" -> style

    def copy(self):
        new_state = copy(self)
        new_state.msgs = {k: v.copy() for k, v in self.msgs.items()}
        return new_state

@dataclass
class ProcessParameters:
    user_id: str
    roles_json_path: str
    llm_model: str
    max_iterations: int

# ====== Cost utilities ======

def compute_llm_cost(solution, state: AgentState):
    """Return a dict with cost breakdown, or None if model not recognized."""
    pricing_by_model = {
        "gpt-5": {"input": 1.250 / 1_000_000, "cached_input": 0.125 / 1_000_000, "output": 10.000 / 1_000_000},
        "gpt-5-mini": {"input": 0.250 / 1_000_000, "cached_input": 0.025 / 1_000_000, "output": 2.000 / 1_000_000},
        "gpt-5-nano": {"input": 0.050 / 1_000_000, "cached_input": 0.005 / 1_000_000, "output": 0.400 / 1_000_000},
    }
    model_pricing = pricing_by_model.get(state.llm_model)
    if model_pricing is None:
        return None

    token_usage = solution.response_metadata.get("token_usage", {})
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    cached_tokens = token_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    cost_prompt = (prompt_tokens - cached_tokens) * model_pricing["input"]
    cost_cached = cached_tokens * model_pricing["cached_input"]
    cost_completion = completion_tokens * model_pricing["output"]
    total_cost = cost_prompt + cost_cached + cost_completion

    return {
        "model": state.llm_model,
        "prompt_tokens": prompt_tokens - cached_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "cost_prompt": cost_prompt,
        "cost_cached": cost_cached,
        "cost_completion": cost_completion,
        "total_cost": total_cost,
    }

def my_print(state: AgentState, marker: str, message: str, role_key: str | None = None, kind: str = "msg"):
    """
    Pretty-print with Rich when available.
    - marker: short prefix like "[gm]:::" or anything; shown in panel title.
    - role_key: selects color style from state.role_styles, else default kind style.
    - kind: one of {'header','msg','cost','debug'} maps to theme styles.
    """
    if state.console is None:
        # Fallback: preserve old behavior
        lines = str(message).splitlines()
        formatted = "\n".join(f"{marker} {line}" for line in lines)
        state.print(formatted)
        return

    style = state.role_styles.get(role_key, state.kind_styles.get(kind, "msg"))
    title_text = Text(marker, style=style)

    # For multi-line messages, use a Panel for readability
    panel = Panel.fit(
        Text(str(message)),
        title=title_text,
        border_style=style
    )
    state.console.print(panel)

def print_rule(state: AgentState, title: str, kind: str = "header"):
    if state.console is None:
        state.print(f"\n==== {title} ====\n")
    else:
        state.console.print(Rule(Text(title, style=state.kind_styles.get(kind, "header"))))

def report_new_approximate_cost(state: AgentState):
    msg = f"${state.approximate_cost:.8f}"
    if state.console is None:
        state.print(f"[COST]::: {msg}")
    else:
        state.console.print(Panel(Text(msg), title=Text("[Total Estimated Cost]:::", style=state.kind_styles.get("cost","cost")),
                                  border_style=state.kind_styles.get("cost","cost")))


# ====== Core agent behaviors (generic) ======
def print_turn_pipeline(state: AgentState):
    if state.console is None:
        state.print(" -> ".join([state.gm_key] + state.player_keys + [state.hist_key]))
        return
    parts = []
    for k in [state.gm_key] + state.player_keys + [state.hist_key]:
        parts.append(Text(k, style=state.role_styles.get(k, "header")))
    # Interleave arrows
    display = Text("")
    for i, part in enumerate(parts):
        if i: display.append("  →  ", style="dim")
        display.append(part)
    state.console.print(Rule(display))


def game_master_generate_solution(state: AgentState) -> AgentState:
    print_turn_pipeline(state)
    print_rule(state, f"Iteration {state.current_iteration} — Game Master")
    my_print(state, f"[{state.gm_key}]:::", f"[iteration {state.current_iteration}] Game Master is responding...",
             role_key=state.gm_key, kind="debug")

    # Turn 0 instruction: create *a plausible destabilizing event* affecting some subset of players.
    user_iter_0 = dedent("""\
        Generate a plausible destabilizing world event for this tabletop exercise.
        It may affect some or all players; scale can be minor or major.
        Examples (non-exhaustive): critical comms outage, order misunderstanding,
        rogue actor, industrial accident, natural disaster, market shock, etc.
        Be realistic and specific.""")

    # Later turns: check equilibrium, else produce a new development; summarize prior player actions.
    if state.current_iteration > 1:
        summary_bits = []
        summary_bits.append(f"Previous Game Master state:\n{state.msgs[state.gm_key][-1]}")
        for k in state.player_keys:
            label = k
            summary_bits.append(f"Previous {label} approach:\n{state.msgs[k][-1]}")
        user_iter_1plus = (
            "\n\nGiven the previous game state and player approaches above:\n"
            "1) Decide if the system reached EQUILIBRIUM: True/False.\n"
            "2) If False, introduce a new destabilizing development (any scale; internal/external).\n"
            "Finally, briefly summarize last round's actions by each player; that plus the equilibrium/development "
            "is the new game state."
        )
        user_content = "\n\n".join(summary_bits) + "\n\n" + user_iter_1plus
    else:
        user_content = user_iter_0

    return agent_generate_solution(
        state=state,
        marker=f"[{state.gm_key}]:::",
        agent_key=state.gm_key,
        user_content=user_content
    )

def player_generate_solution_factory(player_key: str):
    def _fn(state: AgentState) -> AgentState:
        my_print(state, f"[{player_key}]:::", f"[iteration {state.current_iteration}] {player_key} is responding...",
                 role_key=player_key, kind="debug")
        game_state = state.msgs[state.gm_key][-1]
        user_content = (
            f"The current game state (from Game Master) is:\n{game_state}\n\n"
            f"Given your role and objectives, take your turn with concrete actions and optional messages."
        )
        return agent_generate_solution(
            state=state,
            marker=f"[{player_key}]:::",
            agent_key=player_key,
            user_content=user_content
        )
    return _fn

def agent_generate_solution(state: AgentState, marker: str, agent_key: str, user_content: str) -> AgentState:
    my_print(state, marker, f"[iteration {state.current_iteration}] Agent {agent_key} is responding...",
            role_key=agent_key, kind="debug")

    new_state = state.copy()
    llm = ChatOpenAI(model=new_state.llm_model)

    messages = [
        SystemMessage(content=new_state.systems[agent_key]),
        HumanMessage(content=user_content),
    ]

    solution = llm.invoke(messages)

    # Cost (gracefully skip if unknown model)
    cost_details = compute_llm_cost(solution, state)
    if cost_details:
        new_state.approximate_cost += cost_details['total_cost']
    report_new_approximate_cost(new_state)

    # Record
    new_state.msgs[agent_key].append(solution.content)

    my_print(state, marker, f"{solution.content}", role_key=agent_key, kind="msg")
    my_print(state, marker, f"[iteration {state.current_iteration} - DEBUG] Exiting agent_generate_solution.",
             role_key=agent_key, kind="debug")

    return new_state

def should_continue(state: AgentState) -> Literal["continue", "finish"]:
    if state.current_iteration > state.max_iterations:
        state.print(f"[iteration {state.current_iteration} - DEBUG] Reached max_iterations; finishing.")
        return "finish"
    else:
        state.print(f"[iteration {state.current_iteration} - DEBUG] Still under max_iterations; continuing.")
        return "continue"

def increment_iteration(state: AgentState) -> AgentState:
    new_state = state.copy()
    new_state.current_iteration += 1
    state.print(f"[iteration {state.current_iteration} - DEBUG] Iteration incremented to {new_state.current_iteration}")
    return new_state

def historian_generate_solution(state: AgentState) -> AgentState:
    state.print(f"[iteration {state.current_iteration} - DEBUG] Entering historian_generate_solution.")
    print_rule(state, "Summarizing History")
    # Build evolution string
    user_content = ["Evolution of Rounds:\n"]
    rounds = len(state.msgs[state.gm_key])
    for i in range(rounds):
        user_content.append(f"\nIteration {i+1}:")
        user_content.append(f"Game Master:\n{state.msgs[state.gm_key][i]}")
        for k in state.player_keys:
            label = k
            user_content.append(f"{label}:\n{state.msgs[k][i]}")
    user_content.append(dedent("""
        Based on these rounds, provide a historical summary. Be fact-based and coherent; length is fine when warranted.
    """))
    user_content = "\n".join(user_content)

    llm = ChatOpenAI(model=state.llm_model)
    messages = [
        SystemMessage(content=state.systems[state.hist_key]),
        HumanMessage(content=user_content),
    ]

    state.print(f"[iteration {state.current_iteration} - DEBUG] Generating historical account with LLM...")
    historical_account = llm.invoke(messages)
    state.print(f"[iteration {state.current_iteration} - DEBUG] Historical account obtained.")

    new_state = state.copy()
    new_state.historical_account = historical_account.content

    cost_details = compute_llm_cost(historical_account, state)
    if cost_details:
        new_state.approximate_cost += cost_details['total_cost']
    report_new_approximate_cost(new_state)

    state.print(f"[iteration {state.current_iteration} - DEBUG] Exiting historian_generate_solution.")
    return new_state

# ====== Logging & LaTeX ======

def save_as_log(state: AgentState) -> AgentState:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chat_history_log_dir = os.path.join(script_dir, "chat_history_logs")
    os.makedirs(chat_history_log_dir, exist_ok=True)

    user_id = state.user_id
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{user_id}_chat_history_{state.llm_model}_{timestamp_str}"

    # Structured JSON log (replaces brittle regex parsing)
    log_obj = {
        "llm_model": state.llm_model,
        "iterations_completed": len(state.msgs[state.gm_key]),
        "max_iterations": state.max_iterations,
        "approximate_cost": state.approximate_cost,
        "players_order": state.player_keys,
        "roles": {
            "game_master": state.gm_key,
            "historian": state.hist_key,
            "systems": state.systems,  # full system prompts (optional; comment out if you don't want to persist)
        },
        "rounds": []
    }
    role_labels = {state.gm_key: "Game Master", state.hist_key: "Historian"}
    for k in state.player_keys:
        role_labels[k] = k.replace("_"," ").title()
    log_obj["role_labels"] = role_labels

    rounds = len(state.msgs[state.gm_key])
    for i in range(rounds):
        entry = {
            "iteration": i + 1,
            "game_master": state.msgs[state.gm_key][i],
            "players": {k: state.msgs[k][i] for k in state.player_keys}
        }
        log_obj["rounds"].append(entry)

    log_obj["historical_account"] = state.historical_account

    json_filename = os.path.join(chat_history_log_dir, f"{base}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, ensure_ascii=False, indent=2)

    state.print(f"[DEBUG] Chat history JSON saved as '{json_filename}'.")

    new_state = state.copy()
    new_state.chat_history_log_file = json_filename
    return new_state

def parse_log_json(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def sanitize_latex(text: str) -> str:
    text = re.sub(r'[\u2500-\u259F]+', '-', text)
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    regex = re.compile('|'.join(re.escape(k) for k in replacements.keys()))
    text = regex.sub(lambda m: replacements[m.group()], text)
    text = re.sub(r'^\s*###\s*(.+)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*##\s*(.+)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#\s*(.+)$', r'\\section{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
    return text

def render_to_latex(state: AgentState, data: dict, template_path='.', template_name='template_NWiseBears.tex.jinja', output_file='log.tex'):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_name)
    output = template.render(**data)
    output = sanitize_latex(output)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    state.print(f"[DEBUG] LaTeX file '{output_file}' generated successfully.")

def make_latex_from_chat_log(state: AgentState) -> AgentState:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    latex_dir = os.path.join(script_dir, "generated_tex")
    template_path = os.path.join(script_dir, "jinja_templates")
    os.makedirs(latex_dir, exist_ok=True)

    log_data = parse_log_json(state.chat_history_log_file)
    base = os.path.splitext(os.path.basename(state.chat_history_log_file))[0]
    latex_filename = os.path.join(latex_dir, f"{base}.tex")

    render_to_latex(
        state,
        log_data,
        template_path=template_path,
        template_name="template_NWiseBears.tex.jinja",
        output_file=latex_filename
    )

    new_state = state.copy()
    new_state.latex_filename = latex_filename
    state.print(f"[FILE]::: {os.path.basename(latex_filename)}")
    return new_state

# ====== Graph construction (dynamic) ======

def build_agent_graph(player_keys: List[str], gm_key: str, hist_key: str):
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(gm_key, game_master_generate_solution)
    for k in player_keys:
        graph.add_node(k, player_generate_solution_factory(k))
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node(hist_key, historian_generate_solution)
    graph.add_node("save_as_log", save_as_log)
    graph.add_node("make_latex_from_chat_log", make_latex_from_chat_log)

    # Edges: GM -> first player -> ... -> last player -> increment
    if not player_keys:
        raise ValueError("At least one player is required.")
    graph.add_edge(gm_key, player_keys[0])
    for i in range(len(player_keys) - 1):
        graph.add_edge(player_keys[i], player_keys[i + 1])
    graph.add_edge(player_keys[-1], "increment_iteration")

    # Conditional: increment -> (continue ? GM : Historian)
    graph.add_conditional_edges(
        "increment_iteration",
        should_continue,
        {"continue": gm_key, "finish": hist_key}
    )

    # Wrap-up
    graph.add_edge(hist_key, "save_as_log")
    graph.add_edge("save_as_log", "make_latex_from_chat_log")
    graph.add_edge("make_latex_from_chat_log", END)

    graph.set_entry_point(gm_key)
    return graph.compile()

# ====== Runner ======
def _build_console_and_styles(player_keys, gm_key, hist_key):
    # Theme for non-role kinds
    theme = Theme({
        "cost": "bold yellow",
        "debug": "dim",
        "header": "bold",
        "msg": "none",
    })
    console = Console(theme=theme)

    # A pleasant palette; extend if you commonly have many players
    palette = [
        "cyan", "magenta", "green", "blue", "bright_cyan",
        "bright_magenta", "bright_green", "bright_blue", "turquoise4",
        "deep_sky_blue1", "plum1", "spring_green2"
    ]

    # deterministic assignment, stable across runs given order
    role_keys = [gm_key] + player_keys + [hist_key]
    role_styles = {}
    for i, k in enumerate(role_keys):
        if k == gm_key:
            role_styles[k] = "bold bright_white on grey27"
        elif k == hist_key:
            role_styles[k] = "bold bright_white on dark_green"
        else:
            role_styles[k] = f"bold {palette[i % len(palette)]}"

    # Per-kind default styles (can be used when not role-specific)
    kind_styles = {
        "header": "header",
        "msg": "msg",
        "cost": "cost",
        "debug": "debug",
    }
    return console, role_styles, kind_styles


def run_this_thing(params: Dict[str, Any], output_queue: Queue):
    roles = load_roles(params["roles_json_path"])
    gm_key = roles["_gm_key"]
    hist_key = roles["_hist_key"]
    player_keys = roles["_player_keys"]

    # Build systems map and init msg buckets
    systems = {
        gm_key: roles["game_master"]["system"],
        hist_key: roles["historian"]["system"],
    }
    for p, key in zip(roles["players"], player_keys):
        systems[key] = p["system"]

    msgs = {k: [] for k in [gm_key, hist_key] + player_keys}

    agent_graph = build_agent_graph(player_keys, gm_key, hist_key)
    max_iterations = int(params["max_iterations"])

    console, role_styles, kind_styles = _build_console_and_styles(player_keys, gm_key, hist_key)
    state = AgentState(
        systems=systems,
        player_keys=player_keys,
        gm_key=gm_key,
        hist_key=hist_key,
        llm_model=params["llm_model"],
        max_iterations=max_iterations,
        print=output_queue.put,  # remains for non-rich sinks, but we’ll prefer console
        user_id=params["user_id"],
        msgs=msgs,
        console=console,
        role_styles=role_styles,
        kind_styles=kind_styles,
    )

    my_print(state, "[DEBUG]:::", "Invoking the graph...", kind="debug")
    my_print(state, "[DEBUG]:::", f"Using LLM: {params['llm_model']}", kind="debug")
    result = agent_graph.invoke(state, {"recursion_limit": 9999999999999})

    my_print(state, "[DEBUG]:::", "Graph invocation complete.", kind="debug")
    my_print(state, "[DEBUG]:::", result["historical_account"], kind="msg")
    return result

# ====== CLI example ======


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an N-agent, JSON-configured tabletop simulation with Rich output."
    )
    parser.add_argument(
        "-r", "--roles",
        required=True,
        help="Path to roles JSON (e.g., ./src/roles/roles_example.json)",
    )
    parser.add_argument(
        "-u", "--user-id",
        default="DemoUser",
        help="User/session identifier used in logs and filenames (default: DemoUser)",
    )
    parser.add_argument(
        "-m", "--llm-model",
        default="gpt-5-mini",
        help="LLM model name (e.g., gpt-5, gpt-5-mini, gpt-5-nano)",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=2,
        help="Maximum number of iterations/rounds (default: 2)",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.roles):
        parser.error(f"--roles path not found: {args.roles}")

    # Build params dict for run_this_thing (your function currently expects a dict)
    input_params = {
        "user_id": args.user_id,
        "roles_json_path": args.roles,
        "llm_model": args.llm_model,
        "max_iterations": args.iterations,
    }

    class PrintQueue:
        def put(self, item): print(item)

    output_queue = PrintQueue()
    state = run_this_thing(input_params, output_queue)

    from src.main import generate_pdf
    print(f"Generating PDF from: {state['latex_filename']} . . .")
    pdf_path = generate_pdf(state['latex_filename'], use_secure=False)
    print("PDF written to:", pdf_path)
