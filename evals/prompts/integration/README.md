# Prompt integration and trajectory evaluation

This folder is for realistic trajectory replay after component-level suites are
frozen. It must not replace fixed component packets.

Runtime order:

```text
Planner
  -> Controller chooses an action
  -> Reader creates Observations
  -> Checker updates EvidenceMemory
  -> Controller chooses the next action
  -> repeat until Root is sufficient
  -> Answerer receives the question graph and accepted Evidence
```

Recommended evaluation layers:

1. **Component replay:** every Prompt sees a fixed, validated input packet. Use
   this for A/B attribution and regression detection.
2. **Frozen trajectory replay:** downstream Prompts see approved packets sampled
   from real upstream runs. Use this for realism without allowing inputs to
   drift between A/B candidates.
3. **Live end to end:** all candidate Prompts run together. Use this only after
   component candidates are frozen.

For each real trajectory, preserve the exact Planner output, every Controller
input/action, every Reader request/raw output, every Checker input/raw output
and applied state, and the final Answerer input/raw output. Gold and human
reviews remain separate from model-visible packets.
