# Agent Delegation Methodology — Redukt

> Framework for managing multi-agent workflows without context window saturation.

## 1. Core Principles

1. **Context Budget**: Main agent reserves context for implementation + decision-making.
   Research, documentation, and independent evaluations go to sub-agents.
2. **Single Responsibility**: Each agent handles ONE task. Complex tasks split into subtasks.
3. **Parallel by Default**: Independent tasks launch simultaneously. Sequential only when
   output of A is input of B.
4. **Test at Segments**: Not per-task. Group related tasks into segments, test at boundaries.
5. **Metrics Before Code**: Define what to measure BEFORE implementing. Each task declares
   its metric contribution to the overall goal.

## 2. Task Classification

| Category | Context | Agent Type | Example |
|----------|---------|------------|---------|
| Research (web) | Background | Explore/General | Literature review, feasibility study |
| Research (code) | Background | Explore | Codebase analysis, dependency mapping |
| Implementation | Main | N/A | Code changes, refactoring |
| Evaluation (no code) | Background | General | Data analysis, cross-referencing |
| Evaluation (needs code) | Main | N/A | Benchmark runs, model execution |
| Documentation | Background | General | Writing docs, updating changelog |

## 3. Agent Communication Protocol

### Launch
- Provide: task description, success criteria, output format, file paths
- Specify: "research only, no code changes" or "implement and report"
- Set: `run_in_background: true` for independent work

### Results
- Agent returns structured summary (metrics, decisions, file paths)
- Main agent integrates results, makes GO/NO-GO decisions
- Results NOT visible to user — main agent summarizes

### Sub-agent Pattern
- Agents can delegate further via their own Agent tool calls
- Depth limit: 2 levels (main → agent → sub-agent)
- Each level halves available context — keep sub-tasks narrow

## 4. Metrics Capture Protocol

Each task declares:
```
Task: R-XXX
Metric: <what to measure>
Baseline: <current value>
Target: <expected direction>
Capture: <how/where measured>
Contribution: <how this affects the overall goal>
```

Metrics are captured BEFORE and AFTER each task, stored in task output.
At segment boundary: aggregate metrics, identify per-task contribution.

## 5. Segment Testing Protocol

### At segment boundary:
1. `cargo build --release` — zero warnings
2. `cargo test` — all pass
3. BPB reference: `hybrid-eval enwik8 10KB ≈ 1.28`
4. Review per-task metrics: did each task contribute positively?

### Rollback protocol:
- If segment fails: identify which task caused regression
- Revert that task's changes, re-test
- Document regression cause

## 6. Execution Template

```
Phase 1: PARALLEL
  ├── Agent A: [task] → background
  ├── Agent B: [task] → background
  └── Main: [task] → foreground

Phase 2: SEQUENTIAL (depends on Phase 1 results)
  └── Main: [integration task]

CHECKPOINT: build + test + metrics

Phase 3: PARALLEL
  ├── Agent: documentation update
  └── Main: commit + push
```

## 7. Anti-patterns

- ❌ Agent does same search as main (duplication)
- ❌ Agent modifies files that main is also editing (conflicts)
- ❌ Testing per-task instead of per-segment (slow, redundant)
- ❌ Launching agent for trivial tasks (overhead > benefit)
- ❌ Background agent for work that blocks next step (should be foreground)
- ❌ Forgetting to summarize agent results to user (invisible work)
