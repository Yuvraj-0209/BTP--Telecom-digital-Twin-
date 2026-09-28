"""Generate the synthetic training dataset: (reports before, action) -> reports after.

Run:  python make_dataset.py --samples 500
Writes results/dataset.csv (one row per cell per sample) and results/edges.csv
(which cells are neighbours, for a graph neural network).

Inputs are only what a real RIC could see (KPM reports + the action).
User positions stay hidden inside the simulator and are never saved.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sim

OUT = Path(__file__).parent / "results"
CIO_CHOICES_DB = (-6, -4, -2, 0, 2, 4, 6)
KPM_FIELDS = ("prb_util_pct", "RRC.ConnMean", "DRB.UEThpDl", "DRB.UEThpDl_p5", "PEE.AvgPower", "edge_share")


def random_settings(rng, scenario, nonzero_share):
    """Random CIO per cell (mostly 0) and, at night, some sleeping cells."""
    cio = np.where(rng.random(sim.N_CELLS) < nonzero_share, rng.choice(CIO_CHOICES_DB, sim.N_CELLS), 0)
    asleep = rng.random(sim.N_CELLS) < (0.3 if scenario == "night" else 0.0)
    return cio.astype(float), asleep


def random_hotspot(rng):
    """A crowd of random size somewhere in the middle of the network."""
    x, y = rng.uniform(-400, 400, 2)
    return (x, y, int(rng.integers(10, 40)), float(rng.uniform(40, 120)))


def one_sample(sample_id, seed):
    rng = np.random.default_rng(seed)
    scenario = rng.choice(list(sim.SCENARIOS))
    hotspot = random_hotspot(rng) if scenario == "stadium" else None
    _, demand, rx = sim.make_scenario(scenario, rng, hotspot)

    cio_now, asleep_now = random_settings(rng, scenario, nonzero_share=0.1)
    cio_new, asleep_new = random_settings(rng, scenario, nonzero_share=0.3)

    before = sim.kpm_report(sim.simulate(rx, demand, cio_now, asleep_now))
    after = sim.kpm_report(sim.simulate(rx, demand, cio_new, asleep_new))

    for c in range(sim.N_CELLS):
        row = {"sample": sample_id, "seed": seed, "scenario": scenario, "cell": c}
        row["before_asleep"] = int(asleep_now[c])
        row["before_cio_db"] = cio_now[c]
        row.update({f"before_{k}": before[c][k] for k in KPM_FIELDS})
        row["action_asleep"] = int(asleep_new[c])
        row["action_cio_db"] = cio_new[c]
        row.update({f"after_{k}": after[c][k] for k in KPM_FIELDS})     # labels
        yield row


def neighbour_edges():
    """Cells are neighbours if their towers are the same or adjacent."""
    site_dist = np.linalg.norm(sim.SITES[:, None] - sim.SITES[None], axis=2)
    for a in range(sim.N_CELLS):
        for b in range(sim.N_CELLS):
            d = site_dist[sim.CELL_SITE[a], sim.CELL_SITE[b]]
            if a != b and d <= sim.ISD_M * 1.01:
                yield {"cell_a": a, "cell_b": b, "site_distance_m": round(float(d), 1)}


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--first-seed", type=int, default=1000)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    rows = (row for i in range(args.samples) for row in one_sample(i, args.first_seed + i))
    n = write_csv(OUT / "dataset.csv", rows)
    e = write_csv(OUT / "edges.csv", neighbour_edges())
    print(f"Wrote {n} rows ({args.samples} samples x {sim.N_CELLS} cells) to {OUT / 'dataset.csv'}")
    print(f"Wrote {e} neighbour links to {OUT / 'edges.csv'}")
