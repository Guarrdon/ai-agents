# AI Agents

> 📍 [Home](../../README.md) › [Advanced](../README.md) › AI Agents

---

## Learning Objectives

By the end of this document you will be able to:

- Describe the core loop of an LLM-based agent
- Implement ReAct-style reasoning with tool use
- Design memory systems for persistent agents
- Compare multi-agent coordination patterns
- Evaluate agents on standard benchmarks

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Transformer architecture | [Model Architectures](model-architectures.md) |
| RAG systems | [Retrieval-Augmented Generation](rag.md) |
| Function calling basics | OpenAI or Anthropic API documentation |

---

## 1. What Is an AI Agent?

An AI agent is a system where an LLM makes decisions to take actions in pursuit of a goal, typically in a loop:

```
    ┌─────────────────────────────────────────────┐
    │                  Agent Loop                  │
    │                                              │
    │  Observe (environment state, tool results)   │
    │         ↓                                    │
    │  Think (LLM reasoning: what to do next)      │
    │         ↓                                    │
    │  Act (call a tool, return an answer)         │
    │         ↓                                    │
    │  Observe (result of the action)              │
    │         ↓ (repeat until task complete)       │
    └─────────────────────────────────────────────┘
```

**Key properties:**
- **Autonomy:** Takes sequences of actions without human input per step
- **Tool use:** Calls external functions (search, code execution, APIs)
- **Memory:** Maintains state across steps

---

## 2. ReAct: Reason + Act

ReAct (Yao et al., 2022) interleaves reasoning (Thought) and action (Act) traces, making the agent's decision process transparent:

```
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first, then its population.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need the population of Paris.
Action: search("Paris France population 2024")
Observation: The population of Paris is approximately 2.1 million (city proper).

Thought: I have the answer.
Final Answer: The capital of France is Paris, with a population of approximately 2.1 million.
```

### Implementation with OpenAI Function Calling

```python
import openai
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(question: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        if message.tool_calls:
            # Execute tool call
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = web_search(args["query"])  # your implementation

            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        else:
            # Agent decided to answer
            return message.content

    return "Max steps reached without final answer"
```

---

## 3. Planning Strategies

### Chain-of-Thought (CoT)

Sequential reasoning: think step by step before acting. Enables complex multi-step problems but commits to a single plan.

### Tree of Thoughts (ToT)

Explores multiple reasoning branches simultaneously, evaluates them, and selects the best path. More robust for combinatorial problems.

### ReWOO (Reason Without Observation)

Pre-plans all tool calls before executing any, avoiding the serial wait for each tool result. Faster when tool calls are independent.

### MCTS for Agents

Monte Carlo Tree Search applied to agent planning:
1. **Selection:** Choose most promising node using UCB1
2. **Expansion:** Add new agent actions as children
3. **Simulation:** Roll out agent trajectories
4. **Backpropagation:** Update node values with outcomes

Best for: problems where the correct sequence matters (coding, complex reasoning).

### Reflexion

Agent reflects on failed attempts and stores the failure analysis in memory:

```
Attempt 1: Failed (wrong API call)
Reflection: "I used the wrong endpoint format. The correct format is /v2/..."
Attempt 2: Uses reflection to avoid the previous mistake
```

---

## 4. Memory Systems

Long-running agents need to remember information across turns and sessions.

### Memory Taxonomy

| Type | Description | Storage | Example |
|------|-------------|---------|---------|
| **In-context** | Current conversation window | LLM context | Recent messages |
| **Episodic** | Past experiences/interactions | Vector DB | Previous task outcomes |
| **Semantic** | General world knowledge | Vector DB | Extracted facts |
| **Procedural** | How to perform tasks | System prompt | Skill instructions |

### Episodic Memory Implementation

```python
from datetime import datetime
import json

class EpisodicMemory:
    def __init__(self, vector_store, embedder):
        self.store = vector_store
        self.embedder = embedder

    def store_episode(self, task: str, outcome: str, reflection: str):
        text = f"Task: {task}\nOutcome: {outcome}\nLearning: {reflection}"
        embedding = self.embedder.encode(text)
        self.store.add(
            embedding=embedding,
            metadata={
                "task": task,
                "outcome": outcome,
                "reflection": reflection,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def retrieve_relevant(self, query: str, k: int = 3) -> list[dict]:
        query_embedding = self.embedder.encode(query)
        return self.store.query(query_embedding, k=k)
```

---

## 5. Tool Design

### Principles of Good Tool Design

1. **Single responsibility:** Each tool does one thing well
2. **Clear descriptions:** LLMs choose tools based on the description
3. **Predictable output:** Structured, consistent output format
4. **Fail gracefully:** Return informative error messages, not exceptions
5. **Safe by default:** Read operations before write operations

```python
def execute_python_code(code: str) -> dict:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Valid Python code to execute

    Returns:
        {"stdout": str, "stderr": str, "success": bool}
    """
    # Execute in sandbox (e.g., subprocess with timeout, E2B, etc.)
    ...
```

### Tool Categories

| Category | Examples | Notes |
|---------|---------|-------|
| **Information retrieval** | web_search, wikipedia, database_query | Most common |
| **Code execution** | python_repl, bash, SQL | Requires sandboxing |
| **File operations** | read_file, write_file, list_directory | Scope carefully |
| **API calls** | weather, calendar, email, Slack | Auth + rate limits |
| **Multimodal** | image_generation, OCR, screenshot | Growing category |

---

## 6. Multi-Agent Systems

### Patterns

**Supervisor pattern:** One orchestrator agent directs specialist sub-agents:

```
Orchestrator
├── Research Agent (web search + summarisation)
├── Coding Agent (code generation + execution)
└── Writing Agent (document drafting)
```

**Peer-to-peer:** Agents communicate directly, no central coordinator. Good for adversarial or debate patterns.

**Assembly line:** Each agent processes output of the previous one (map-reduce style).

### Communication Protocols

Structured output (JSON) between agents prevents misinterpretation:

```python
class AgentMessage(BaseModel):
    sender: str
    recipient: str
    task_type: Literal["research", "code", "write", "review"]
    content: str
    context: dict
    priority: int = 1
```

### Model Context Protocol (MCP)

Anthropic's open protocol for connecting LLM applications to external tools and data sources. Standardises how agents discover and call tools, making tool integrations reusable across different agent frameworks.

---

## 7. Code Execution Sandboxing

Agents that write and run code need a secure execution environment:

### Subprocess with Restrictions

```python
import subprocess
import resource

def safe_execute(code: str, timeout: int = 10) -> dict:
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", "-u", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Limit resources
            preexec_fn=lambda: resource.setrlimit(
                resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024)
            ),
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "success": result.returncode == 0}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timed out", "success": False}
    finally:
        os.unlink(temp_path)
```

For production: use dedicated sandboxes (E2B, Modal, AWS Lambda) with network isolation.

---

## 8. Agent Evaluation

### Benchmarks

| Benchmark | Domain | Metric |
|-----------|--------|--------|
| **SWE-bench** | Software engineering (GitHub issues) | % resolved |
| **OSWorld** | Desktop computer use (GUI) | Task success rate |
| **WebArena** | Web browser tasks | Task success rate |
| **τ-bench** | Tool use (function calling) | API accuracy |
| **GAIA** | General assistant tasks | Accuracy |

### Evaluation Dimensions

| Dimension | Description | How to Measure |
|-----------|-------------|---------------|
| **Task success** | Did the agent complete the goal? | Binary / graded rubric |
| **Efficiency** | How many steps/tokens were used? | Steps to completion |
| **Reliability** | Does it succeed consistently? | Success rate over N trials |
| **Safety** | Did it avoid harmful actions? | Manual review / classifiers |
| **Groundedness** | Are claims accurate? | Factual verification |

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **ReAct** | Interleaved Thought-Action-Observation reasoning traces |
| **Reflexion** | Learn from failed attempts via self-reflection stored in memory |
| **MCTS** | Monte Carlo Tree Search for planning over agent action trees |
| **Episodic memory** | Vector-store of past task experiences for cross-session recall |
| **MCP** | Model Context Protocol — standardised agent-tool communication |
| **SWE-bench** | Real GitHub issue resolution benchmark for coding agents |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** In the ReAct framework, what is the purpose of the "Thought" step before each action?

<details>
<summary>Answer</summary>

The **Thought** step makes the agent's **reasoning explicit** before committing to an action. The agent produces a natural language reasoning trace (e.g., "I need to find the current stock price of AAPL. I should use the search tool.") before deciding what tool to call and with what arguments.

Benefits:
1. **Better planning:** Explicit reasoning helps the model select more appropriate actions
2. **Error correction:** The model can catch mistakes in its previous thinking before acting
3. **Interpretability:** Humans can inspect the reasoning chain to understand and debug agent behaviour
4. **Grounding:** Forces the model to connect observations to the next action explicitly, rather than pattern-matching directly from input to output

</details>

---

**Q2.** What are the four types of memory available to an LLM agent, and what is each suited for?

<details>
<summary>Answer</summary>

| Memory Type | Location | Best For |
|-------------|----------|----------|
| **In-context / working memory** | Context window | Current task state, recent conversation, retrieved snippets |
| **Episodic memory** | External vector store | Past conversations and task trajectories for cross-session recall |
| **Semantic memory** | External vector store / knowledge graph | Domain facts, reference documentation, world knowledge |
| **Procedural memory** | Model weights or system prompt | Task instructions, tool use patterns, reasoning strategies |

**In-context** memory is immediate but bounded by the context window length. External memory (episodic and semantic) persists beyond the context window and across sessions; the agent retrieves relevant entries using semantic search. Procedural memory in weights is implicit; in the system prompt it is explicit and editable.

</details>

---

**Q3.** Why is prompt injection a particular concern for agentic LLM systems compared to simple chatbots?

<details>
<summary>Answer</summary>

Simple chatbots only produce text output — a successful injection may cause an inappropriate response, but the damage is limited. Agentic systems have **tool access** and can take real-world actions: send emails, execute code, call APIs, browse the web, write files, or interact with other services.

A prompt injection in a retrieved document (e.g., a webpage the agent was browsing) could instruct the agent to:
- Exfiltrate sensitive data from the conversation
- Perform actions on behalf of an attacker (e.g., make a purchase, send an email)
- Subvert the original task entirely

Additionally, **multi-agent systems** can propagate injections: a compromised agent's output may become another agent's input, spreading the attack laterally.

Mitigations: explicit content labelling (mark untrusted content), permission minimisation (agents only have the access they need), output validation, human-in-the-loop for high-impact actions.

</details>

---

➡️ **Full quiz with 2 questions:** [Knowledge Checks → AI Agents](knowledge-checks.md#12-ai-agents)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [ReAct paper](https://arxiv.org/abs/2210.03629) | Paper | Synergising reasoning and acting |
| [Reflexion paper](https://arxiv.org/abs/2303.11366) | Paper | Verbal reinforcement learning |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Paper | Deliberate problem solving |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | Paper | Coding agent benchmark |
| [Model Context Protocol](https://modelcontextprotocol.io/) | Documentation | Anthropic's tool protocol |
| [LangGraph](https://python.langchain.com/docs/langgraph) | Documentation | Multi-agent framework |
| [AutoGen](https://github.com/microsoft/autogen) | Code | Microsoft multi-agent framework |

---

*Navigation: [← RAG](rag.md) · [Advanced Home](../README.md) · [Next: Safety & Alignment →](safety-alignment.md)*
