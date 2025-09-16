# 3 Wise Bears
3 Simple AI Agents Conversing

This project contains a small collection of *very simple* AI agents based
on langchain and langgraph that communicate in a round-robin fashion, attempting
to improve their answer from the critique / actions of the other agents.

## What is This?
There are basically two entry points to demonstration agents - a web-interface
to `goldilocks_and_the_3_agents.py` and a command-line interface to
`N_wise_bears.py`.  These are briefly explained below.

### Web Interface to 3 Agents
This is an easy to understand / approachable demo that has some canned
personalities for 3 agents that go around and round on a problem and critique
the ideas in different ways.  "Agent 1" is the one generating a creative
solution to the posed problem and the other 2 agents critique it in different
ways.  Each agent in turn sees the critique of the previous one and tries
to take that into account in their next iteration (e.g. improvement).
The user can control the number of times it iterates.

**NOTE** The interface provides choices for GPT-5 nano, GPT-5 mini, and GPT-5
each increasing in cost.  We generally find that GPT-5 nano is incapable
of this agentic exchange, so the user may want to change that to GPT-5.  An
advanced user would likely change this to an on-premises model such as 
OpenAI's OSS 20b or 120b models.  Again, this is just a demonstration of a
technique.  GPT-5 mini is a good deal cheaper and faster if it works.

GPT-5 iterating in this way can take a while - the user should be patient.
In our experience we see roughtly 90-120 seconds for each agent to respond
and approximate the cost at about $1 for 3 iterations in this way.

Once complete, another agent will summarize the process, try to write it
out as TeX and compile it into a PDF for easy viewing of the conversation.
This *usually* works (LLMs aren't great at generating TeX yet) but look
in `src/generated_tex` for chat history logs, `.tex` and hopefully
a `.pdf` - with time stamps.

**NOTE** Consider watching debug output on the Developer Console in your
web browser to see messages like:

```
. . .
[Agent 3]::: [iteration 1 - DEBUG] Exiting agent_generate_solution. script.js:100:15
[iteration 1 - DEBUG] Iteration incremented to 2 script.js:100:15
[iteration 2 - DEBUG] Reached max_iterations; finishing.
. . .
```

### N Wise Bears - CLI
The `N_wise_bears.py` is a command line interface (CLI) that more or less
does the same thing but phrases the problem more as a game master and
multiple players taking turns.  Each turn the game master essentially
changes the game state in some way and then simultaneously (different from
the Goldilocks example which is sequentially) each player takes a turn.

There are text files that control some example inputs to the game
master's role and personalities for the players which you can edit
to your liking.

## Environment Setup
Using `environment.yml`:

```
conda env create -f environment.yml
conda activate 3_wise_bears
```

### Dependencies
The generated TeX will be compiled into a PDF with `xelatex`.  Most TeX installations
have this already, but you should check with `xelatex --version` and install it
if you don't have it.

### LLM API Key
Generally speaking, you'll need an LLM API key to use this.  As delivered,
this assumes the OpenAI API, but this can easily be changed to a local LLM
if you desire.  You'll need your API setup something like this:

```
export OPENAI_API_KEY="sk-..."
```

## Running the Demos
Running of the two demos provided in this package are explained below.
### Web Interface to 3 Agents
1. `python -m src.main` - note the URL to connect, it will probably
say ` * Running on http://127.0.0.1:5000`
2. Connect a web browser to the above address
3. Select your parameters, initially we'd suggest: GPT-5 mini, 
2 iterations, and the dinosaur extinction problem.
4. Press `Launch` button and watch the colored frames under `Agent Output` 
for outputs.  In about ~10 minutes a final answer will be provided and a
button to download a PDF.

Worked?  Try a different problem, or your own problem.  Try changing the
"Voice" of each of the agents.

### N Wise Bears - CLI
TODO


## Notice of Copyright Assertion (O4903):
© 2025. Triad National Security, LLC. All rights reserved.  This program was
produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC
for the U.S.  Department of Energy/National Nuclear Security Administration.
All rights in the program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to
reproduce, prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.

Copyright 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

 

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
