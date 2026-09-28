# CLAUDE.md: BTP project, AI-Driven Digital Twin for 5G Networks

Project context for Claude Code. Read this whole file before doing anything in this repo.
Last updated: 28 September 2026. Update the "Decisions" and "Open questions" sections whenever something changes.

---

## 1. Who we are and how to work with us

- Team of 3, B.Tech CSE final year (NSUT Delhi, 2023–2027 batch). This is our BTP (final-year project).
- Laptops: **2 Macs + 1 Windows laptop**. Every command, path and script must work on all three.
- We are **new to telecom**. Explain concepts from scratch with an everyday analogy before using jargon.
- Supervisor's condition: she co-authors and helps publish in a **reputed journal** only if the work is **unique and useful**. Her literature rule: use **2026 journal papers** (not conferences) in the literature review.

### How Claude should behave here
- **Be skeptical and double-check.** We are not always right, and neither are you. Research before asserting; say what you checked.
- **Give a confidence score** (e.g. "Confidence: 80%") on every non-trivial answer, suggestion or claim.
- **Say no when needed.** Decline out-of-scope work, push back on unclear requirements, refuse approaches that break the design rules below, and ask instead of assuming. When declining: say so clearly, give the reason briefly, and offer an alternative or say what you need.
- **Never call anything "new" or "first"** without checking it against Section 9 (competitors) and a fresh arXiv search. New preprints in this area appear every month.
- Keep replies casual, direct and short. Lead with the answer.

### Coding principles (non-negotiable)
- **Touch only what you must.** Clean up only your own mess. No drive-by refactors.
- **Minimum code that solves the problem.** Nothing speculative, no "future-proofing".
- **DRY:** every piece of knowledge has one authoritative place (constants live in `sim.py`, scenario definitions in `sim.SCENARIOS`).
- **KISS** and **least astonishment:** plain functions, obvious names, no clever tricks.
- **When a bug is reported:** do NOT start by fixing it. First write a test that reproduces it (in `tests/`, pytest). Then fix, and show the test passing.
- Ask before adding any new dependency.

---

## 2. The project in one paragraph

We build a **digital twin** of a small 5G network: a fast AI copy that answers "what happens if I change this network setting?" before the setting is applied. Our research is about **what to do when the AI copy is unsure**. Existing work either refuses to act (OpenTwin) or falls back to an old setting (Almohammedi et al.). We instead **spend a small, fixed budget of slow, accurate simulations** on exactly the candidate actions whose predicted range is ambiguous, and show this recovers most of the lost benefit while keeping rule violations within budget, including under sudden traffic surges.

- **Working title:** Uncertainty-Gated, Budget-Aware Escalation for Digital-Twin-Based RAN Control under Non-Stationary Traffic
- **Research question:** Can conformal-uncertainty-gated escalation let a fast ML twin make load-balancing and energy-saving decisions nearly as good as full simulation, at a fraction of the compute, without increasing QoS violations under unseen traffic surges?
- **Target venues (tentative):** IEEE Access, Elsevier Computer Communications, Elsevier Computer Networks, IEEE Networking Letters. Stretch: IEEE TNSM.
- **Status (28 Sept 2026):** literature review done, direction not yet approved by supervisor, Python prototype simulator built.

---

## 3. Domain glossary (plain words)

| Term | Plain meaning |
|---|---|
| UE | A user's phone |
| gNB | A 5G base station (the radio box + antennas on a tower) |
| Site / cell | A tower (site) usually has 3 antenna panels, each covering ~120°. Each panel's area is a **cell**. Our network: 7 sites × 3 = 21 cells |
| RAN | Radio Access Network: all towers and their radio equipment. The part we control |
| PRB | Physical Resource Block, one unit of radio capacity ("a bank counter"). 20 MHz at 30 kHz spacing = 51 PRBs; 100 MHz = 273 |
| PRB utilisation | % of PRBs busy. Around 90–95%+ means the cell is jammed |
| RSRP / SINR | Signal strength ("bars") / signal quality over noise and interference |
| Handover | A phone switching to a neighbour cell, usually when the neighbour is better by a margin (A3 event) |
| **CIO** | Cell Individual Offset (dB). Bias that makes a cell look better or worse, so **edge users** move. The real knob behind "move users from A to B" |
| Cell sleep | Switching off a spare cell at low traffic to save energy |
| SON | Self-Organizing Networks: older built-in autopilot rules. A baseline to beat |
| O-RAN | Open RAN: open, standard interfaces so third-party software can control the RAN |
| RIC | RAN Intelligent Controller: "the brain computer" that runs control apps |
| Near-RT RIC / xApp | Fast brain, decisions every 10 ms – 1 s. Apps on it are **xApps** |
| Non-RT RIC / rApp | Slow brain (inside the SMO management layer), decisions > 1 s, typically minutes. Apps on it are **rApps**. **Our method is an rApp** |
| E2 | Interface between the near-RT RIC and base stations (CU/DU = "E2 nodes") |
| KPM | E2SM-KPM, Key Performance Measurement: the "report card" counters a cell sends up (PRB usage, connected users, throughput…) |
| RC | E2SM-RC, RAN Control: the "command" direction |
| A1 / O1 | A1: policies from non-RT RIC to near-RT RIC. O1: management/configuration interface |
| Digital twin | A copy of *one specific* network, continuously synced from its reports, used to test actions before applying them |
| Surrogate / fast guesser | ML model predicting "if I take action X, what KPIs will I get?" |
| Conformal prediction | Turns a model's point prediction into a range with a coverage guarantee (e.g. truth inside 90% of the time) |
| ACI | Adaptive Conformal Inference: widens/narrows the range online when misses increase (drift) |
| Escalation | Sending an "unsure" candidate to the slow accurate simulator |
| Budget B | Max number of slow simulator checks per decision |
| Regret | How much worse the chosen action was than the best possible one |

---

## 4. System architecture

There is **no real network**. A simulator plays the real network. Three roles:

```
 Sim A  --KPM reports-->  Controller (RIC + our rApp)  --"what if X?"-->  Digital twin
 (pretend real network)          |                                       - state guess from reports
   ^                             |                                       - fast guesser (GNN) + conformal range
   +-------- command ------------+                                       - slow checker = Sim B
```

| Piece | Real-world name | In our code |
|---|---|---|
| Sim A | Physical network | The simulator with the hidden "true" user positions |
| Link | E2 (KPM up, control down) | Plain Python: `sim.kpm_report(...)` up, `sim.simulate(..., cio_db, asleep)` down. **No real E2/RIC software** |
| Controller | RIC running our rApp | A Python loop, one iteration per control period. **The research method lives here** |
| State guess | Twin synchronisation | Rebuilds a scenario from KPM reports only (user counts per cell, random positions) |
| Fast guesser | Surrogate | Trained ML model + conformal ranges |
| Slow checker | High-fidelity simulation | Sim B: same simulator software, **started from the state guess, never from Sim A's hidden truth** |

### One decision (pseudocode)
```python
reports = sim_A.get_reports()                       # KPM-style, per cell
state   = twin.sync(reports)                        # twin's guess of the current state
safe, unsure = [], []
for action in candidate_actions():                  # CIO / sleep combinations
    low, high = twin.guesser.predict_range(state, action)
    if high <= LIMIT:  safe.append(action)          # clearly fine
    elif low > LIMIT:  continue                     # clearly bad
    else:              unsure.append(action)        # range straddles the rule
for action in most_promising(unsure, budget=B):     # OUR CONTRIBUTION
    if twin.checker.simulate(state, action) <= LIMIT:
        safe.append(action)
chosen = best(safe) if safe else KEEP_CURRENT
sim_A.apply(chosen); outcome = sim_A.run_one_period()
twin.guesser.recalibrate(outcome)                   # ACI update from misses
```
- `most_promising`: rank unsure candidates by potential gain (their optimistic bound vs the best safe option's pessimistic bound); check the top B.
- If nothing is safe: keep the current settings.
- Every period is decided fresh. No lookup table of past decisions.

### Why the twin is unsure (the core of the research)
The controller sees the state **now**; it must predict the state **after** the action. Unknowns: how many users sit near the border (CIO only moves edge users); moved users have weak signal so they cost more PRBs in the new cell; interference coupling between cells; traffic still changing; situations unlike the training data (surges). "Unsure" = the predicted range crosses a rule limit.

### Timing
- Framed at the **non-RT / rApp** timescale (decisions every few minutes), so a slow check of seconds to a minute fits. Report **wall-clock time per decision**.
- In experiments Sim A **pauses between control periods** (turn-based). In real life the network keeps running on the old settings while checks run.

---

## 5. Hard design rules (refuse to break these)

1. **The twin's inputs are only what a real report could contain** (KPM-level counters + the action). Never feed exact user positions, the random seed, or the scenario name to the guesser or the controller. The simulator's hidden truth is used **only to produce labels**.
2. **Sim B must not use Sim A's hidden truth.** It starts from the twin's state guess with its own random draws. Otherwise escalation is "peeking at the answer" and results are meaningless.
3. **Tune thresholds and hyperparameters on validation data, never on test data.**
4. **Split data by simulation run (sample/seed), never by row.**
5. **Keep surges (and outages) out of training** so the test really is "never seen before".
6. Every generated row records `seed` so it can be regenerated. Results must be reproducible from seeds.
7. Don't make RL or an LLM agent the core method. Don't make forecasting the contribution.

---

## 6. Repository layout and commands

```
sim.py            simulator (Sim A / Sim B): layout, users, signal, load, CIO, sleep, power, KPM reports
demo.py           what-if demos: stadium surge (CIO sweep) and 3 am (cell sleep); prints tables, saves charts
make_dataset.py   synthetic dataset: (before KPMs, action) -> after KPMs, one row per cell per sample
README.md         install/run instructions and simplifications (for humans)
results/          generated outputs (charts, dataset.csv, edges.csv); safe to regenerate
```
Planned (not yet written): `tests/` (pytest), `twin/` (state guess, guesser, conformal), `controller.py` (decision loop + baselines), `experiments/`.

### Setup
- Python 3.10+, `numpy`, `matplotlib`. Planned ML stack: PyTorch, PyTorch Geometric, MAPIE, XGBoost (ask before adding).
- Mac: `python3 -m pip install numpy matplotlib`
- Windows: Python from python.org with "Add to PATH", then `py -m pip install numpy matplotlib`. Use `py` instead of `python3`.

### Run
```
python3 demo.py                        # ~2 s; tables + results/stadium_loads.png, stadium_map.png, night_power.png
python3 make_dataset.py --samples 500  # ~5 s; results/dataset.csv (10,500 rows) + results/edges.csv
```

### Cross-platform rules
- Use `pathlib` for paths, `open(..., newline="")` for CSV.
- **ASCII only** in printed output and source (Windows consoles can crash on symbols like ≤ or →).
- matplotlib must use the `Agg` backend in scripts (no GUI needed).

---

## 7. The prototype simulator (`sim.py`)

A snapshot, system-level model. Built 28 Sept 2026 as a prototype for the first teacher demo.

**Models:**
- 7 sites on a hexagon, ISD 500 m, 3 sectors each (azimuths 30°, 150°, 270°) = 21 cells
- Path loss: 3GPP TR 38.901 UMa NLOS formula `13.54 + 39.08 log10(d3D) + 20 log10(fc)` at 3.5 GHz, 6 dB log-normal shadowing per user–site link, min distance 35 m, BS 25 m, UE 1.5 m
- Antenna: 8 dBi, 65° beamwidth, 30 dB front-to-back, horizontal pattern only
- 20 MHz per cell, 51 PRBs, 46 dBm, noise figure 9 dB
- Association: each user joins the awake cell with max (received power + cell CIO)
- Load coupling: interference from a neighbour scales with its load; fixed-point iteration (80 steps, damping 0.5)
- Spectral efficiency: `clip(0.6 * log2(1 + SINR), 0.05, 5.5)` bit/s/Hz; user demand 0.5–1.5 Mbit/s
- Overloaded cell (load > 1): every user gets `demand / load`
- Power: EARTH-style, awake `130 + 4.7 × 40 × load` W, asleep 75 W (approximate values)
- Scenarios (`sim.SCENARIOS`): `normal` (35 users/site), `stadium` (35/site + 30-user crowd at (180 m, 0 m), spread 80 m), `night` (8/site)

**Key functions:** `make_scenario(name, rng, hotspot=None)`, `simulate(rx_dbm, demand, cio_db, asleep)`, `kpm_report(result)`, `summary(result)`.

**KPM report fields per cell** (dotted names follow 3GPP TS 28.552, as used by E2SM-KPM; `prb_util_pct` and `edge_share` are our own derived fields): `RRU.PrbUsedDl`, `RRU.PrbAvailDl`, `prb_util_pct`, `RRC.ConnMean`, `DRB.UEThpDl`, `DRB.UEThpDl_p5` (derived from the `DRB.UEThpDlDist` distribution), `PEE.AvgPower`, `edge_share` (share of users whose best other cell is within 3 dB, from UE measurement reports), `asleep`.

**Known simplifications (state them in any presentation or paper):**
- users stand still within a control period; no packet-level or protocol detail
- CIO is one offset per cell, not per neighbour pair; no handover hysteresis or time-to-trigger
- Shannon-based throughput, not real MCS tables
- no wraparound: edge cells get less interference than in reality
- power model values are rough; the ~22% night saving in the demo is likely optimistic

**Demo results (seed 3), for reference:**
- Stadium: cell 0 at 111% load, 86.5% of users fully served. CIO −2 dB on cell 0 → worst cell 76%, 100% served. CIO −6 dB or more overloads neighbour cell 8. This is the "moving users can overload the neighbour" effect.
- Night: 15 of 21 cells can sleep within the rules (2.86 → 2.22 kW). 18 asleep breaks coverage (25% of users below −5 dB SINR).

---

## 8. Dataset and models

### What is trained
| Component | Trained? | Notes |
|---|---|---|
| Simulator (Sim A/B) | **No**, configured | Physics + rules; the "data factory", the "real world" and the "checker" |
| Fast guesser | **Yes** | Supervised regression. Baselines first: "no change", simple load-transfer formula, XGBoost; then GNN (PyTorch Geometric, e.g. GraphSAGE/GAT) |
| Conformal ranges | Calibrated, not trained | Split conformal on a calibration set + ACI online. Plain split conformal gives equal-width ranges everywhere, so use an **adaptive-width variant** (CQR or normalised residuals) |
| Decision logic | Written rules | Our contribution |
| State guess | Written rules | Reports → scenario for Sim B |
| Traffic forecaster | Optional | Seasonal-naive at most. Not a contribution |

With 21 cells, XGBoost may beat the GNN. That is fine: report honestly. The contribution is the escalation logic, which works with any guesser.

### Labels
Automatic: each simulator run gives `(before reports, action) → (after reports)`. The after-values are the labels (numbers, so regression). Predict the numbers and apply the rules afterwards (not a safe/unsafe classifier), so changing a limit needs no retraining.

### `dataset.csv` columns (one row per cell per sample)
- `sample`, `seed`, `scenario`, `cell`: bookkeeping. **Never feed `scenario` or `seed` to a model.**
- `before_asleep`, `before_cio_db`, `before_<kpm>`: inputs
- `action_asleep`, `action_cio_db`: inputs (the action)
- `after_<kpm>`: labels
- `<kpm>` ∈ `prb_util_pct, RRC.ConnMean, DRB.UEThpDl, DRB.UEThpDl_p5, PEE.AvgPower, edge_share`
- `edges.csv`: neighbour links (`cell_a, cell_b, site_distance_m`) for the graph

### Splits (by sample, never by row)
~60% train (normal + daily ramps), ~10% validation, ~15% calibration, ~15% test (**surges, outages, unseen crowd patterns**; touched once).

### Size
Start with ~2,000 samples; plot a learning curve (500 / 1,000 / 2,000) and grow only if error is still dropping. Confidence in "2,000 is enough": low (~45%).

---

## 9. Competitors and novelty (from our literature notes; verify before citing)

| When the twin is unsure… | Work | Weakness we exploit |
|---|---|---|
| Always trust it | COMIX (IEEE Access 2025) | Regret explodes under drift |
| Fall back to last-known-good | Almohammedi et al. 2026, arXiv 2607.22857 (preprint) | Stale under surges; reactive; single knob, one cell; "homogeneous twin" (same model + injected offset); threshold tuned on evaluation data |
| Refuse to act (null action) | OpenTwin 2026, arXiv 2605.24662 (preprint) | Keeps only ~40% of perfect-foresight savings; deliberately avoids per-candidate simulation; one action type (sleep) |
| Always run full simulation | Hu et al., arXiv 2512.20640 | Safe but expensive |
| **Budgeted escalation of ambiguous candidates** | **Ours** | Target: most of the benefit at bounded compute, violations ≤ budget |

**Already taken (do not claim):** trust check before acting with conformal (OpenTwin); error monitoring + fallback (Almohammedi); always-simulate (Hu); conformal what-if KPI analysis (Hou et al. IEEE TCCN 2025; Qchohi et al. 2026); rule-based pre-action safety verification (Jawad et al., IEEE Access 2026); twin forecasting (saturated).

**Honest caveat:** conformal routing from a cheap model to an expensive oracle exists in general ML (e.g. CADS arXiv 2605.16401, Cost-Aware ACI arXiv 2605.24463). Our novelty is **application/systems**: the oracle is a network simulator judging counterfactual RAN actions under drift. Fits applied journals, not theory venues.

**Still to read before finalising claims:** VeraRAN (arXiv 2608.01047), xTRUCE (arXiv 2608.28532), EuCNC 2026 "Towards Trustworthy 6G Network Digital Twins…". Re-search arXiv monthly ("conformal digital twin budget escalation").

**Backup angles if scooped:** (B) surge-robust conformal gating (ACI vs change-point-reset vs windowed conformal); (C) coupled multi-cell CIO + sleep with conformal risk control.

Key numbers (from papers, per our notes): Almohammedi normalised regret 0.017 vs 0.159 COMIX-style; OpenTwin 6.12% energy saved gated vs 11.1% unguarded; Reiss et al. 63–96% of oracle savings; arXiv 2509.10097 ~13% saving (51 cells).

---

## 10. Evaluation plan

**Baselines:** (1) always-trust twin (COMIX-style), (2) EWMA + last-known-good (Almohammedi), (3) conformal gate with null action (OpenTwin-style, = B=0), (4) always-simulate (Hu-style, = B=all), (5) rule-based SON thresholds, (6) oracle (exhaustive on ground truth).

**Metrics:** regret / % of oracle benefit **vs simulator calls per decision** (the main plot); QoS-violation rate vs target α; energy saved as % of oracle; coverage and range width over time around surge onsets; wall-clock per decision; surrogate error per KPI.

**Rules (example defaults, not fixed by any paper):** every awake cell ≤ 90% PRB utilisation; no users below −5 dB SINR; slowest users keep a minimum rate. Knobs: CIO ∈ {−6,…,+6} dB in 2 dB steps; sleep on/off; transmit power only for the outage scenario.

**Scenarios:** stadium surge (CIO load balancing), 3 am low traffic (sleep), site outage (neighbour compensation), stationary + daily ramps.

**Rigour:** 20+ seeds with 95% confidence intervals; ablations (no escalation, no ACI, vary B); thresholds tuned on validation only; code released on GitHub.

---

## 11. Decisions made

- **MATLAB is out** (NSUT likely has no licence; a 30-day trial is too short). Octave cannot run MATLAB's 5G Toolbox.
- **No real RIC/E2 stack.** Emulate KPM reports and commands in Python.
- **Frame as a non-RT rApp.**
- **Python prototype simulator exists** (Section 7) and is used for early development and demos.
- The twin must differ from "reality" (Section 5, rule 2).

## 12. Open questions

1. **Long-term simulator:** ns-3 + 5G-LENA (if someone can do C++) vs a validated Python simulator. Possible stronger setup: ns-3 as Sim A, Python as data factory and checker (a "heterogeneous twin"), at roughly double the build work.
2. Can the supervisor accept arXiv preprints as baselines (both closest competitors are preprints)?
3. Does "2026 journals only" apply to all references or just the literature-review table?
4. Letter vs full journal.
5. Supervisor approval of the refined research question.

## 13. Verified tool facts (checked 27–28 Sept 2026)

- **5G-LENA** (ns-3 NR module, CTTC, GPLv2): NR-v5.0 (3 July 2026) added CIO handover biasing and a handover overhaul; NR-v5.1 added more handover fixes. Tested on Ubuntu 22.04/25.10, Arch, and macOS Sequoia (Apple Silicon); compatible with ns-3.48. Windows not listed, so use WSL2 + Ubuntu 22.04. No energy/sleep power model in its release notes.
- **ns-O-RAN** (Northeastern/Mavenir) builds on the ns-3 **mmWave** module, not 5G-LENA; implements E2SM-KPM and E2SM-RC.
- **Simu5G** (OMNeT++, C++, LGPL): X2 handover, inter-cell interference; CIO support unverified.
- **Sionna SYS** (NVIDIA, Python): link adaptation, power control, scheduling; no handover, so CIO would be hand-written.
- **Datasets:** Telecom Italia Milan (Nov–Dec 2013, 10-minute slots, ODbL) for daily traffic shape; OpenCellID (CC BY-SA 4.0) for real tower positions (no azimuth, power or load).
- The dotted KPM counter names used here (`RRU.*`, `RRC.*`, `DRB.*`, `PEE.*`) appear in srsRAN's E2SM-KPM metric definitions (3GPP TS 28.552).

## 14. Timeline (tight, front-loaded)

| When | Milestone |
|---|---|
| Late Sept – early Oct 2026 | Supervisor approval, tools set up, simulator decision |
| Weeks 1–3 | Tier A/B reading, paper matrix, toy conformal + GNN exercises |
| Weeks 4–7 (to mid-Nov) | Simulator + dataset, 5 baselines reproduced, first regret plot |
| Mid-Nov – Dec | Guesser + conformal + escalation + main plot + stress scenarios (must exist by end of Dec) |
| Jan – Mar 2027 | Seeds, ablations, novelty re-check, writing |
| Mar – Apr 2027 | Submission |

## 15. Do NOT

- Claim "first digital twin for X" or "first to check trust before acting".
- Use NVIDIA AODT or ray tracing (too heavy).
- Connect to any real operator network.
- Tune on test data, or split by row.
- Let the guesser or controller see user positions, seeds or scenario names.
- Let Sim B start from Sim A's hidden state.
- Add dependencies, frameworks or abstraction layers without asking.
- Submit to predatory journals; verify Scopus / Web of Science indexing.
