# BTP network simulator (prototype)

A small Python simulator of a 5G network: 7 towers x 3 cells = 21 cells, a few hundred phones.
It plays "the real network" (Sim A) in our digital-twin project and produces the synthetic dataset.

## Install (once per laptop, about 5 minutes)

You need Python 3.10 or newer, plus two libraries.

**Mac**
```
python3 -m pip install numpy matplotlib
```

**Windows**
1. Install Python from https://www.python.org/downloads/ and tick "Add python.exe to PATH" during install.
2. In Command Prompt:
```
py -m pip install numpy matplotlib
```

## Run

From inside this folder (on Windows use `py` instead of `python3`):
```
python3 demo.py                       # what-if tables + 3 charts in results/
python3 make_dataset.py --samples 500 # synthetic dataset in results/ (about 5 seconds)
```

## Files

| File | What it does |
|---|---|
| `sim.py` | The simulator: towers, users, signal, load, CIO, sleep, power, KPM reports |
| `demo.py` | Stadium surge (try CIO values) and 3 am (try sleeping cells); prints tables, saves charts |
| `make_dataset.py` | Runs many random situations and random actions; saves before / action / after per cell |
| `results/` | Output charts and CSV files |

## What it models

- Signal loss: 3GPP TR 38.901 urban macro (non line-of-sight) formula at 3.5 GHz, 6 dB random shadowing
- Antennas: 3 sectors per tower, 65 degree beamwidth
- Each user joins the cell with the strongest signal after adding that cell's CIO
- Each cell has 20 MHz (51 PRBs); a user's share depends on its signal quality (SINR)
- Cells are coupled: a busy cell causes more interference to its neighbours
- Sleeping cells serve nobody and cause no interference
- Power: EARTH-style model (idle power + load-dependent part; lower power when asleep)

## Simplifications (say these when presenting)

- Snapshot model: users stand still during one control period; no packet-level detail
- CIO is one offset per cell, not per neighbour pair; no handover hysteresis or timers
- Throughput from a simplified Shannon formula, not real 5G MCS tables
- No towers outside the 7 simulated, so edge cells get less interference than in reality
- Power model numbers are approximate

It is a prototype for showing the idea. Before the paper, we either validate it against
published numbers (e.g. Almohammedi et al.) or move the "real network" role to ns-3 + 5G-LENA.

## Dataset columns

One row per cell per sample.
- `before_*`: what the cell reported before the action (inputs)
- `action_cio_db`, `action_asleep`: the action (inputs)
- `after_*`: what the cell reported after the action (labels to predict)
- `sample`, `seed`, `scenario`: for splitting the data and regenerating any row. Do not feed `scenario` to the model, a real RIC wouldn't know it.

`edges.csv` lists which cells are neighbours, for building the graph for a GNN.
