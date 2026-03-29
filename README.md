# MiniSociety

`MiniSociety` is a small multi-agent reinforcement learning simulation of a reputation-driven society.
Agents choose whether to interact, whether to accept proposals, and whether to cooperate or defect once matched.
They learn from direct experience, remember past behavior, and can exchange limited third-party trust information.

## Current Model

- Environment: iterated social dilemma with partner selection.
- Learning: decentralized PPO in PyTorch.
- Memory: each agent tracks per-agent reputation and confidence.
- Observation: each agent sees its own beliefs about others, incoming proposals, and how others currently see it.
- Reputation sharing: direct interaction updates always apply; third-party trust sharing only happens after mutual cooperation.

Default payoff matrix:

- Mutual cooperate: `(100, 100)`
- Defect / cooperate: `(103, -106)`
- Cooperate / defect: `(-106, 103)`
- Mutual defect: `(-2, -2)`
- Isolation: `-1`

## Requirements

- Python 3.10+
- `torch`
- Optional:
  - `matplotlib` for visualization
  - `PySide6` for the GUI

Example install:

```bash
pip install torch matplotlib PySide6
```

## Run

Terminal training run:

```bash
python marl_minisociety.py
```

GUI:

```bash
python marl_minisociety.py --gui
```

Final-episode visualization:

```bash
python marl_minisociety.py --visualize
```

Checkpoint save/load:

```bash
python marl_minisociety.py --save-checkpoint minisociety.pt
python marl_minisociety.py --load-checkpoint minisociety.pt
```

## Tests

```bash
python -m py_compile marl_minisociety.py tests/test_marl_minisociety.py
python -m unittest discover -s tests -v
```

## Repo Layout

- `marl_minisociety.py`: environment, PPO agents, CLI, GUI, visualization
- `tests/test_marl_minisociety.py`: regression tests
