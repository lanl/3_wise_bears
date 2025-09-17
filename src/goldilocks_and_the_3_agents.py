import sys
from copy import copy
from queue import Queue
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal, Callable, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import time

from textwrap import dedent
import json
import ast

import subprocess
import os
from datetime import datetime

import re
from jinja2 import Environment, FileSystemLoader
import sys
import os

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


# Define our state schema
@dataclass
class AgentState:
    agent1_system: str
    agent1_voice: str
    agent2_system: str
    agent2_voice: str
    agent3_system: str
    agent3_voice: str
    llm_model: str
    max_iterations: int
    question: str

    agent1_msgs: List[str] = field(default_factory=list)  # we're hard coding this as 3 agents...
    agent2_msgs: List[str] = field(default_factory=list)  # these are the messages from these agents
    agent3_msgs: List[str] = field(default_factory=list)  # for record keeping
    approximate_cost: float = 0.0
    chat_history_log_file: str = ""
    current_iteration: int = 0
    final_answer: str = "" # Final refined answer
    latex_filename: str =""
    print: Callable[[Any], None] = print # default to standard print function

    def copy(self):
        # Create a shallow copy of the current instance
        new_state = copy(self)
        
        # Create new lists for mutable attributes to avoid sharing references
        new_state.agent1_msgs = self.agent1_msgs.copy()
        new_state.agent2_msgs = self.agent2_msgs.copy()
        new_state.agent3_msgs = self.agent3_msgs.copy()
        
        return new_state

@dataclass
class ProcessParameters:
    question: str
    agent1_system: str
    agent1_voice: str
    agent1_custom: str
    agent2_system: str
    agent2_voice: str
    agent2_custom: str
    agent3_system: str
    agent3_voice: str
    agent3_custom: str
    llm_model: str
    iterations: int



def compute_llm_cost(solution, state):
    """
    Calculate the approximate cost of an LLM transaction using token usage
    and a pricing structure indexed by model name.

    Args:
        solution: The result object from llm.invoke() containing response_metadata.
        state: A dictionary with application state, including the key 'llm_model'.

    Returns:
        A dictionary with detailed cost breakdown.
    """
    # this'll be awkward, but we'll have to find a way to keep this updated... i don't know of a way to
    # programmatically query these values.
    # there's a discussion here: https://community.openai.com/t/get-model-pricing-using-the-api/862940
    # saying that this is a commonly requested feature, but not provided...
    # i'm getting the model pricing from here: https://openai.com/api/pricing/
    pricing_by_model = {
        "gpt-5": {
            "input": 1.250 / 1_000_000,        # $1.250 per 1M tokens
            "cached_input": 0.125 / 1_000_000, # $0.125 per 1M tokens
            "output": 10.000 / 1_000_000,      # $10.000 per 1M tokens
        },
        "gpt-5-mini": {
            "input": 0.250 / 1_000_000,        # $0.250 per 1M tokens
            "cached_input": 0.025 / 1_000_000, # $0.025 per 1M tokens
            "output": 2.000 / 1_000_000,       # $2.000 per 1M tokens
        },
        "gpt-5-nano": {
            "input": 0.050 / 1_000_000,        # $0.050 per 1M tokens
            "cached_input": 0.005 / 1_000_000, # $0.005 per 1M tokens
            "output": 0.400 / 1_000_000,       # $0.400 per 1M tokens
        }
    }

    # Retrieve model name from state and get its pricing info
    model_name = state.llm_model
    model_pricing = pricing_by_model.get(model_name)
    if model_pricing is None:
        raise ValueError(f"No pricing information available for model: {model_name}")

    # Extract token usage details from the solution's metadata
    token_usage = solution.response_metadata.get("token_usage", {})
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    # Check if there are any cached prompt tokens
    cached_tokens = token_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    # Calculate costs based on token usage and pricing
    cost_prompt = (prompt_tokens - cached_tokens) * model_pricing["input"]
    cost_cached = cached_tokens * model_pricing["cached_input"]
    cost_completion = completion_tokens * model_pricing["output"]
    total_cost = cost_prompt + cost_cached + cost_completion

    # Return the breakdown as a dictionary
    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens - cached_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "cost_prompt": cost_prompt,
        "cost_cached": cost_cached,
        "cost_completion": cost_completion,
        "total_cost": total_cost,
    }


def print_llm_cost_details(state: AgentState, cost_details, marker):
    """
    Print the cost breakdown in a formatted manner.

    Args:
        cost_details: Dictionary containing the detailed cost breakdown.
    """
    my_print(state, marker, f"Model: {cost_details['model']}")
    my_print(state, marker, f"Prompt Tokens (non-cached): {cost_details['prompt_tokens']}")
    my_print(state, marker, f"Cached Prompt Tokens: {cost_details['cached_tokens']}")
    my_print(state, marker, f"Completion Tokens: {cost_details['completion_tokens']}")
    my_print(state, marker, f"Cost for prompt (non-cached): ${cost_details['cost_prompt']:.8f}")
    my_print(state, marker, f"Cost for cached tokens: ${cost_details['cost_cached']:.8f}")
    my_print(state, marker, f"Cost for completion tokens: ${cost_details['cost_completion']:.8f}")
    my_print(state, marker, f"Total approximate cost: ${cost_details['total_cost']:.8f}")


def report_new_approximate_cost(state: AgentState):
    my_print(state, "[COST]:::", f"{state.approximate_cost:.8f}")


def my_print(state: AgentState, marker, message):
    """
    Prepends the given marker (including delimiter) to every line of the message.
    For example, if marker is "[Agent 1]:::" and message is multiline,
    each line will be formatted as "[Agent 1]::: <line>".
    """
    # Split the message into lines.
    lines = message.splitlines()
    # Prepend marker to each line and join them with newline.
    formatted_message = "\n".join(f"{marker} {line}" for line in lines)
    state.print(formatted_message)


def agent1_generate_solution(state: AgentState) -> AgentState:
    user_iter_1plus = ""
    if state.current_iteration > 0:
        user_iter_1plus += f"\nPrevious solution: {state.agent1_msgs[-1]}"
        user_iter_1plus += f"\nCritique: {state.agent2_msgs[-1]}"
        user_iter_1plus += f"\nCompetitor perspective: {state.agent3_msgs[-1]}"
        user_iter_1plus += (
            "\n\n**You must explicitly list how this new solution differs from the previous solution,** "
            "point by point, explaining what changes were made in response to the critique and competitor perspective."
            "\nAfterward, provide your updated solution."
        )

    new_state = agent_generate_solution(
        state=state,
        marker="[Agent 1]:::",
        agent_number=1,
        agent_str="agent1",
        user_iter_0="Research this problem and generate a solution.",
        user_iter_1plus=user_iter_1plus,
    )

    return new_state

def agent2_generate_solution(state: AgentState) -> AgentState:
    solution = state.agent1_msgs[-1]
    user_content = (
        f"Question: {state.question}\n"
        f"Proposed solution: {solution}\n"
        "Provide a detailed critique of this solution. Identify potential flaws, assumptions, and areas for improvement."
    )

    new_state = agent_generate_solution(
        state=state,
        marker="[Agent 2]:::",
        agent_number=2,
        agent_str="agent2",
        user_iter_0=user_content,
        user_iter_1plus="", # no special instructions for >0 for this agent
    )

    return new_state

def agent3_generate_solution(state: AgentState) -> AgentState:
    solution = state.agent1_msgs[-1]
    critique = state.agent2_msgs[-1]

    user_content = (
        f"Question: {state.question}\n"
        f"Proposed solution: {solution}\n"
        f"Critique: {critique}\n"
        # "Simulate how a competitor, government agency, or other stakeholder might respond to this solution."
    )

    new_state = agent_generate_solution(
        state=state,
        marker="[Agent 3]:::",
        agent_number=3,
        agent_str="agent3",
        user_iter_0=user_content,
        user_iter_1plus="", # no special instructions for >0 for this agent
    )

    return new_state


def agent_generate_solution(state: AgentState,
                            marker,
                            agent_number,
                            agent_str,
                            user_iter_0,
                            user_iter_1plus) -> AgentState:

    # we're 0-indexing like good computer folk, but to print to the user we'll +1 it so that
    # they see something more human-like
    my_print(state, marker, f"[iteration {state.current_iteration + 1}] Agent {agent_number} is responding . . .")

    current_iter = state.current_iteration
    user_content = f"Question: {state.question}\n"

    agent_voice = getattr(state, f"{agent_str}_voice")
    if agent_voice != "None":
        user_content += f"You are role playing as this person: {agent_voice}.  Your response *MUST* be in this voice."

    if current_iter > 0:
        user_content += user_iter_1plus
    else:
        user_content += user_iter_0

    # we're going to modify the state, so we need to make a copy and we'll return the new state at teh eend
    new_state = state.copy()

    # Initialize LLM
    # sadly w/ these things, every call to the LLM is stateless, so we have to do it new each time
    # and pump in any state we want it to remember from previous messages / conversations.
    llm = ChatOpenAI(model=new_state.llm_model)


    # Provide a system message to define this agent's role
    messages = [
        SystemMessage(content=getattr(new_state, f'{agent_str}_system')),
        HumanMessage(content=user_content),
    ]

    solution = llm.invoke(messages)

    # this is the full solution struct returned - basically useful for debugging only
    # my_print(state, marker, f"{solution}")

    # compute the approximate costs of that operation
    cost_details = compute_llm_cost(solution, state)
    new_state.approximate_cost += cost_details['total_cost']
    # this is a verbose printing of the LLM costs, we generally don't need to see all of this
    # print_llm_cost_details(state, cost_details, marker)
    report_new_approximate_cost(new_state)
    # my_print(state, marker, f"New Approx Total Cost: {new_state.approximate_cost}")

    # append this solution to our record keeping / state
    getattr(new_state, f'{agent_str}_msgs').append(solution.content)

    my_print(state, marker, f"{solution.content}")
    my_print(state, marker, f"[iteration {state.current_iteration} - DEBUG] Exiting agent_generate_solution.")
    return new_state


def should_continue(state: AgentState) -> Literal["continue", "finish"]:
    if state.current_iteration >= state.max_iterations:
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


def generate_final_answer(state: AgentState) -> AgentState:
    """Generate the final, refined answer based on all iterations."""
    state.print(f"[iteration {state.current_iteration} - DEBUG] Entering generate_final_answer.")
    prompt = f"Original question: {state.question}\n\n"
    prompt += "Evolution of solutions:\n"

    for i in range(state.max_iterations):
        prompt += f"\nIteration {i + 1}:\n"
        prompt += f"Agent 1: {state.agent1_msgs[i]}\n"
        prompt += f"Agent 2: {state.agent2_msgs[i]}\n"
        prompt += f"Agent 3: {state.agent3_msgs[i]}\n"

    prompt += "\nBased on this iterative process, provide the final, refined solution. You must respond using only ASCII characters."

    # Initialize LLM
    llm = ChatOpenAI(model=state.llm_model)

    state.print(f"[iteration {state.current_iteration} - DEBUG] Generating final answer with LLM...")
    final_answer = llm.invoke(prompt)
    state.print(f"[iteration {state.current_iteration} - DEBUG] Final answer obtained.")


    new_state = state.copy()
    new_state.final_answer = final_answer.content

    # compute the approximate costs of that operation
    cost_details = compute_llm_cost(final_answer, state)
    new_state.approximate_cost += cost_details['total_cost']
    # this is a verbose printing of the LLM costs, we generally don't need to see all of this
    # print_llm_cost_details(state, cost_details, marker)
    report_new_approximate_cost(new_state)

    state.print(f"[iteration {state.current_iteration} - DEBUG] Exiting generate_final_answer.")
    return new_state

def save_as_log(state: AgentState) -> AgentState:
    # Compute the absolute path from the directory containing this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chat_history_log_dir = os.path.join(script_dir, "chat_history_logs")

    if not os.path.exists(chat_history_log_dir):
        os.makedirs(chat_history_log_dir)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(
        chat_history_log_dir,
        f"chat_history_{state.llm_model}_{timestamp_str}.txt"
    )

    with open(filename, "w", encoding="utf-8") as file:
        file.write("=== Agent Interaction Log ===\n\n")

        # Metadata
        file.write(f"Question: {state.question}\n")
        file.write(f"LLM Model: {state.llm_model}\n")
        file.write(f"Iterations: {state.current_iteration} / {state.max_iterations}\n")
        file.write(f"Approximate Cost: {state.approximate_cost:.8f}\n")

        # Iterative Dialogue
        for i, (solution, critique, perspective) in enumerate(
                zip(state.agent1_msgs, state.agent2_msgs, state.agent3_msgs),
                start=1
        ):
            file.write(f"--- Iteration {i} ---\n")
            file.write(f"Agent 1 Solution:\n{solution}\n\n")
            file.write(f"Agent 2 Critique:\n{critique}\n\n")
            file.write(f"Agent 3 Perspective:\n{perspective}\n\n")

        # Final Summary
        file.write("=== Final Answer ===\n")
        file.write(f"{state.final_answer}\n\n")


    state.print(f"[DEBUG] Chat history log successfully saved as '{filename}'.")

    new_state = state.copy()
    new_state.chat_history_log_file = filename

    return new_state



def parse_log(filename: str):
    """Parse the structured text log into a Python dictionary."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract metadata
    meta = {}
    meta['question'] = re.search(r"Question: (.+)", content).group(1)
    meta['llm_model'] = re.search(r"LLM Model: (.+)", content).group(1)
    meta['current_iteration'], meta['max_iterations'] = re.search(
        r"Iterations: (\d+) / (\d+)", content).groups()
    meta['approximate_cost'] = re.search(r"Approximate Cost: (.+)", content).group(1)

    iteration_pattern = re.compile(
        r"--- Iteration \d+ ---\s*"
        r"Agent 1 Solution:\n(?P<solution>.+?)\n(?=Agent 2 Critique:)\s*"
        r"Agent 2 Critique:\n(?P<critique>.+?)\n(?=Agent 3 Perspective:)\s*"
        r"Agent 3 Perspective:\n(?P<perspective>.+?)(?=(--- Iteration|\=\=\= Final Answer|\Z))",
        re.DOTALL
    )

    iterations = []
    for match in iteration_pattern.finditer(content):
        iterations.append({
            'solution': match.group('solution').strip(),
            'critique': match.group('critique').strip(),
            'perspective': match.group('perspective').strip()
        })

    final_answer_match = re.search(r"\=\=\= Final Answer \=\=\=\n(.+?)(?:\n\=\=\=|\Z)", content, re.DOTALL)
    meta['final_answer'] = final_answer_match.group(1).strip() if final_answer_match else "(No final answer provided)"

    summary_match = re.search(r"\=\=\= Summary Report \=\=\=\n(.+?)(?:\n\=\=\=|\Z)", content, re.DOTALL)
    meta['summary_report'] = summary_match.group(1).strip() if summary_match else "(No summary report provided)"

    meta['iterations'] = iterations
    return meta


def sanitize_latex(text: str) -> str:
    """Escape special LaTeX characters and convert markdown-like headings."""
    replacements = {
        # '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        # '{': r'\{',
        # '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }

    # Escape special LaTeX characters first
    regex = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    text = regex.sub(lambda match: replacements[match.group()], text)

    # Convert markdown-style headings to LaTeX sections:
    text = re.sub(r'^\s*###\s*(.+)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*##\s*(.+)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#\s*(.+)$', r'\\section{\1}', text, flags=re.MULTILINE)

    # Convert markdown bold (**text**) to LaTeX bold (\textbf{})
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)

    # Convert markdown italics (*text*) to LaTeX italics (\textit{})
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)

    return text


def render_to_latex(state: AgentState, data: dict, template_path='.', template_name='template_3agents.tex.jinja', output_file='log.tex'):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_name)
    output = template.render(**data)
    output = sanitize_latex(output)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    state.print(f"[DEBUG] LaTeX file '{output_file}' generated successfully.")


def make_latex_from_chat_log(state: AgentState) -> AgentState:
    # Compute absolute paths from the directory containing this file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    latex_dir = os.path.join(script_dir, "generated_tex")
    template_path = os.path.join(script_dir, "jinja_templates")

    if not os.path.exists(latex_dir):
        os.makedirs(latex_dir)

    log_filename = state.chat_history_log_file
    log_data = parse_log(log_filename)

    base = os.path.splitext(os.path.basename(log_filename))[0]
    latex_filename = os.path.join(latex_dir, f"{base}.tex")

    render_to_latex(
        state,
        log_data,
        template_path=template_path,  # use the absolute path
        template_name="template_3agents.tex.jinja",
        output_file=latex_filename
    )

    new_state = state.copy()
    new_state.latex_filename = latex_filename
    state.print(f"[FILE]::: {base}.tex")

    return new_state


def build_agent_graph():
    # Initialize the graph
    graph = StateGraph(
        AgentState,
    )

    # Add nodes
    graph.add_node("agent1", agent1_generate_solution)
    graph.add_node("agent2", agent2_generate_solution)
    graph.add_node("agent3", agent3_generate_solution)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("finalize", generate_final_answer)
    graph.add_node("save_as_log", save_as_log)
    graph.add_node("make_latex_from_chat_log", make_latex_from_chat_log)

    # Add simple edges for the known flow
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    # After agent3 is done, we **always** go to increment_iteration
    graph.add_edge("agent3", "increment_iteration")

    # Then from increment_iteration, we have a conditional:
    # If we 'continue', we go back to agent1
    # If we 'finish', we jump to the finalize node
    graph.add_conditional_edges(
        "increment_iteration",
        should_continue,
        {
            "continue": "agent1",
            "finish": "finalize"
        }
    )

    graph.add_edge("finalize", "save_as_log")
    graph.add_edge("save_as_log", "make_latex_from_chat_log")
    graph.add_edge("make_latex_from_chat_log", END)

    # Set the entry point
    graph.set_entry_point("agent1")

    return graph.compile()


def run_this_thing(data: ProcessParameters, output_queue: Queue):
    
    # Create the graph
    agent_graph = build_agent_graph()

    max_iterations = int(data['iterations'])
                                               # add this back in if we add custom voices back in
    agent1_voice_choice = data['agent1_voice'] #if agent1_voice != 'custom' else agent1_custom
    agent2_voice_choice = data['agent2_voice'] #if agent2_voice != 'custom' else agent2_custom
    agent3_voice_choice = data['agent3_voice'] #if agent3_voice != 'custom' else agent3_custom

    # Initialize the state
    state = AgentState(
        agent1_system=data['agent1_system'],
        agent1_voice=agent1_voice_choice,
        agent2_system=data['agent2_system'],
        agent2_voice=agent2_voice_choice,
        agent3_system=data['agent3_system'],
        agent3_voice=agent3_voice_choice,
        llm_model=data['llm_model'],
        max_iterations=max_iterations,
        print=output_queue.put,
        question=data['question'],
    )

    state.print("[DEBUG] Invoking the graph...")
    state.print(f"[DEBUG] Using LLM: {data['llm_model']}")

    # Run the graph
    result = agent_graph.invoke(state, {"recursion_limit": 9999999999999})

    # Print the final answer
    state.print("[DEBUG] Graph invocation complete.")
    state.print(result["final_answer"])

    return result