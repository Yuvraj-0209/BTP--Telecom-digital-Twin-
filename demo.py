"""Two what-if demos on the simulator: a stadium surge and a quiet 3 am.

Run:  python demo.py
Prints what-if tables and saves three charts into the results/ folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim

OUT = Path(__file__).parent / "results"
LIMIT_PCT = 90.0

INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, AQUA, ORANGE, NEUTRAL = "#2a78d6", "#1baf7a", "#eb6834", "#c9c8c2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def is_safe(result):
    s = sim.summary(result)
    return s["worst_cell_util_pct"] <= LIMIT_PCT and s["poor_signal_users_pct"] == 0


def print_table(title, rows):
    print(f"\n{title}")
    header = f"{'action':<34}{'worst cell %':>13}{'fully served %':>16}{'poor signal %':>15}{'power W':>9}  verdict"
    print(header)
    print("-" * len(header))
    for label, result in rows:
        s = sim.summary(result)
        verdict = "SAFE" if is_safe(result) else "breaks a rule"
        print(f"{label:<34}{s['worst_cell_util_pct']:>13}{s['users_fully_served_pct']:>16}"
              f"{s['poor_signal_users_pct']:>15}{s['total_power_w']:>9.0f}  {verdict}")


def stadium_demo():
    rng = np.random.default_rng(3)
    xy, demand, rx = sim.make_scenario("stadium", rng)
    before = sim.simulate(rx, demand)
    stadium_cell = int(before["load"].argmax())

    rows = []
    for offset in range(0, 11, 2):
        cio = np.zeros(sim.N_CELLS)
        cio[stadium_cell] = -offset
        label = "do nothing" if offset == 0 else f"stadium cell CIO -{offset} dB"
        rows.append((label, sim.simulate(rx, demand, cio)))
    print_table(f"STADIUM SURGE: cell {stadium_cell} is overloaded. What if we push edge users away?", rows)

    print("\nWhat the RIC sees for the 3 busiest cells (KPM report, before any action):")
    for rep in sorted(sim.kpm_report(before), key=lambda r: -r["prb_util_pct"])[:3]:
        print("  ", rep)

    good = rows[1][1]    # -2 dB
    too_much = rows[4][1]  # -8 dB
    plot_loads([("Do nothing", before), ("CIO -2 dB: fixed", good), ("CIO -8 dB: pushed too far", too_much)])
    plot_map(xy, before, good, stadium_cell)


def night_demo():
    rng = np.random.default_rng(3)
    _, demand, rx = sim.make_scenario("night", rng)
    base = sim.simulate(rx, demand)
    quietest_first = np.argsort(base["load"])

    rows = []
    for k in (0, 4, 8, 12, 15, 18):
        asleep = np.zeros(sim.N_CELLS, bool)
        asleep[quietest_first[:k]] = True
        rows.append((f"sleep the {k} quietest cells" if k else "all cells on", sim.simulate(rx, demand, asleep=asleep)))
    print_table("3 AM: very few users. How many cells can sleep to save power?", rows)
    plot_power(rows)


def plot_loads(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(13, 3.8), sharey=True)
    cells = np.arange(sim.N_CELLS)
    for ax, (title, result) in zip(axes, cases):
        load_pct = 100 * result["load"]
        colors = [ORANGE if v > LIMIT_PCT else BLUE for v in load_pct]
        ax.bar(cells, load_pct, width=0.7, color=colors)
        ax.axhline(LIMIT_PCT, color=INK_2, lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(cells[::2])
        ax.set_xlabel("cell number")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, max(130, 100 * max(r["load"].max() for _, r in cases) + 10))
    axes[0].set_ylabel("load (% of cell capacity)")
    axes[0].text(20.5, LIMIT_PCT + 3, "90% limit", ha="right", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["within limit", "over 90% (users slowed down)"], loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "stadium_loads.png", dpi=150)
    plt.close(fig)


def plot_map(xy, before, after, stadium_cell):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    was_stadium = before["serving"] == stadium_cell
    for ax, (title, result) in zip(axes, [("Before: do nothing", before), ("After: stadium cell CIO -2 dB", after)]):
        in_stadium = result["serving"] == stadium_cell
        moved = was_stadium & ~in_stadium
        other = ~in_stadium & ~moved
        ax.scatter(*xy[other].T, s=9, color=NEUTRAL, label="other users")
        ax.scatter(*xy[in_stadium].T, s=12, color=BLUE, label=f"served by stadium cell {stadium_cell}")
        if moved.any():
            ax.scatter(*xy[moved].T, s=14, color=AQUA, label="moved to a neighbour cell")
        ax.scatter(*sim.SITES.T, marker="^", s=90, color=INK, label="tower (3 cells each)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("metres")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    axes[0].set_ylabel("metres")
    fig.tight_layout()
    fig.savefig(OUT / "stadium_map.png", dpi=150)
    plt.close(fig)


def plot_power(rows):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [label.replace("sleep the ", "").replace(" quietest cells", " asleep") for label, _ in rows]
    power_kw = [sim.summary(r)["total_power_w"] / 1000 for _, r in rows]
    colors = [BLUE if is_safe(r) else ORANGE for _, r in rows]
    bars = ax.bar(labels, power_kw, width=0.6, color=colors)
    for bar, value, (_, r) in zip(bars, power_kw, rows):
        note = f"{value:.2f} kW" if is_safe(r) else f"{value:.2f} kW\nbreaks a rule"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, note, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("total network power (kW)")
    ax.set_title("3 am: power saved by putting cells to sleep", loc="left", fontsize=11)
    ax.set_ylim(0, max(power_kw) * 1.25)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "night_power.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    stadium_demo()
    night_demo()
    print(f"\nCharts saved in {OUT}")
