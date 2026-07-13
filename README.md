# ADP Experiments

This repository contains small, reproducible experiment scripts around the
Agent Data Protocol (ADP) data release.

Current contents:

- `openhands_sdk_training/`: reproduce the original Qwen3.5 0.8B
  LLaMA-Factory run and track the current Qwen3.5 4B/9B OpenHands SDK
  condenser SFT experiments.
- `codescout_rl/`: smoke-test notes for Codescout RL through Platoon/AReaL on
  a small Qwen3 model.

## Local Workspace Convention

Keep source code and generated experiment artifacts separate:

- Code checkouts and experiment scripts: `~/work/adp/`
- Data, caches, logs, model outputs, and generated artifacts: `~/exp/adp/`

Recommended experiment layout:

```text
~/work/adp/
├── adp-experiments/
└── agent-data-protocol/

~/exp/adp/
├── datasets/
├── runs/
├── cache/
└── tmp/
```

The current experiment ledger is
`openhands_sdk_training/CURRENT_EXPERIMENTS.md`.
