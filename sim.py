"""Tiny system-level 5G simulator for the BTP digital-twin prototype.

It plays the role of "the real network" (Sim A). Given where users are, how much
data they want, and the network settings (CIO per cell, which cells sleep), it
returns what each cell would report to the RIC (KPM-style counters).

Simplifications (say these out loud when presenting):
- snapshot model: users stand still for one control period, no packet-level detail
- CIO is one offset per cell (applied to every user evaluating that cell),
  not one offset per neighbour pair
- no handover hysteresis or time-to-trigger; users simply join the best biased cell
- throughput from attenuated Shannon, not real MCS tables
"""
import numpy as np

# ---- Network layout and radio constants (3GPP TR 38.901 UMa-like) ----
ISD_M = 500.0                    # distance between neighbouring sites
BS_HEIGHT_M, UE_HEIGHT_M = 25.0, 1.5
FREQ_GHZ = 3.5
BW_HZ = 20e6
N_PRB = 51                       # 20 MHz at 30 kHz subcarrier spacing
TX_POWER_DBM = 46.0
ANT_GAIN_DBI, BEAMWIDTH_DEG, FRONT_BACK_DB = 8.0, 65.0, 30.0
SHADOW_STD_DB = 6.0
NOISE_MW = 10 ** ((-174 + 10 * np.log10(BW_HZ) + 9.0) / 10)   # 9 dB noise figure
SECTOR_AZIMUTHS_DEG = (30.0, 150.0, 270.0)
POOR_SINR_DB = -5.0              # below this a user would likely drop the connection

# ---- EARTH-style power model (Auer et al. 2011), approximate macro values ----
P0_W, DELTA_P, PMAX_W, PSLEEP_W = 130.0, 4.7, 40.0, 75.0

_ring = np.deg2rad(30 + 60 * np.arange(6))
SITES = np.vstack([[0.0, 0.0], ISD_M * np.c_[np.cos(_ring), np.sin(_ring)]])  # (7, 2)
CELL_SITE = np.repeat(np.arange(len(SITES)), 3)                               # (21,)
CELL_AZ = np.tile(SECTOR_AZIMUTHS_DEG, len(SITES))                            # (21,)
N_CELLS = len(CELL_SITE)


def drop_users(rng, users_per_site=40, hotspot=None, demand_mbps=(0.5, 1.5)):
    """Place users uniformly around every site, plus an optional crowd.

    hotspot: (x_m, y_m, n_users, spread_m), e.g. a stadium.
    Returns user positions (n, 2) and the data rate each one wants (n,) in Mbit/s.
    """
    n = users_per_site * len(SITES)
    radius = ISD_M / np.sqrt(3) * np.sqrt(rng.random(n))
    angle = rng.uniform(0, 2 * np.pi, n)
    xy = SITES[np.arange(n) % len(SITES)] + np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    if hotspot is not None:
        x, y, crowd, spread = hotspot
        xy = np.vstack([xy, rng.normal([x, y], spread, size=(crowd, 2))])
    return xy, rng.uniform(*demand_mbps, len(xy))


SCENARIOS = {
    # name: (users per site, crowd or None)
    "normal": (35, None),
    "stadium": (35, (180.0, 0.0, 30, 80.0)),   # 30 extra users around (180 m, 0 m)
    "night": (8, None),
}


def make_scenario(name, rng, hotspot=None):
    """Drop users for a named scenario. Returns positions, demand, received power."""
    users_per_site, default_hotspot = SCENARIOS[name]
    xy, demand = drop_users(rng, users_per_site, hotspot or default_hotspot)
    return xy, demand, received_power_dbm(xy, rng)


def received_power_dbm(xy, rng):
    """Power (dBm) every user hears from every cell: shape (n_users, 21)."""
    to_user = xy[:, None, :] - SITES[None, :, :]
    d2d = np.maximum(np.linalg.norm(to_user, axis=2), 35.0)
    d3d = np.hypot(d2d, BS_HEIGHT_M - UE_HEIGHT_M)
    path_loss = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(FREQ_GHZ)   # UMa NLOS
    path_loss += rng.normal(0, SHADOW_STD_DB, d2d.shape)                  # per user-site
    bearing = np.degrees(np.arctan2(to_user[..., 1], to_user[..., 0]))
    off_axis = (bearing[:, CELL_SITE] - CELL_AZ + 180) % 360 - 180
    antenna = ANT_GAIN_DBI - np.minimum(12 * (off_axis / BEAMWIDTH_DEG) ** 2, FRONT_BACK_DB)
    return TX_POWER_DBM + antenna - path_loss[:, CELL_SITE]


def simulate(rx_dbm, demand_mbps, cio_db=None, asleep=None, iterations=80):
    """Run one control period and return per-user and per-cell results.

    Cells are coupled: a busier neighbour transmits more often, so it causes
    more interference, so everyone near it needs more PRBs (load coupling).
    """
    cio_db = np.zeros(N_CELLS) if cio_db is None else np.asarray(cio_db, float)
    asleep = np.zeros(N_CELLS, bool) if asleep is None else np.asarray(asleep, bool)
    awake = ~asleep
    users = np.arange(len(rx_dbm))

    serving = np.where(awake, rx_dbm + cio_db, -np.inf).argmax(axis=1)
    rx_mw = 10 ** (rx_dbm / 10) * awake
    signal = rx_mw[users, serving]

    load = 0.5 * awake
    for _ in range(iterations):
        busy = np.minimum(load, 1.0)
        interference = rx_mw @ busy - signal * busy[serving]
        sinr = signal / (interference + NOISE_MW)
        spectral_eff = np.clip(0.6 * np.log2(1 + sinr), 0.05, 5.5)   # bit/s/Hz
        prb_share = demand_mbps * 1e6 / (BW_HZ * spectral_eff)        # share of the cell each user needs
        load = 0.5 * load + 0.5 * np.bincount(serving, prb_share, N_CELLS)

    throughput = demand_mbps / np.maximum(1.0, load[serving])       # overloaded cells share out
    return {
        "serving": serving, "sinr_db": 10 * np.log10(sinr), "throughput_mbps": throughput,
        "demand_mbps": demand_mbps, "load": load, "asleep": asleep, "rx_dbm": rx_dbm,
    }


def kpm_report(result, edge_margin_db=3.0):
    """What each cell reports to the RIC, using 3GPP TS 28.552 counter names.

    Only aggregate numbers: this is all the digital twin is allowed to see.
    """
    serving, rx, asleep = result["serving"], result["rx_dbm"], result["asleep"]
    awake_rx = np.where(asleep, -np.inf, rx)
    own = awake_rx[np.arange(len(rx)), serving]
    awake_rx[np.arange(len(rx)), serving] = -np.inf
    near_border = own - awake_rx.max(axis=1) < edge_margin_db   # from UE measurement reports

    busy = np.minimum(result["load"], 1.0)
    reports = []
    for c in range(N_CELLS):
        mine = serving == c
        thr = result["throughput_mbps"][mine]
        reports.append({
            "cell": c,
            "asleep": bool(asleep[c]),
            "RRU.PrbUsedDl": int(round(busy[c] * N_PRB)),
            "RRU.PrbAvailDl": N_PRB,
            "prb_util_pct": round(float(100 * busy[c]), 1),
            "RRC.ConnMean": int(mine.sum()),
            "DRB.UEThpDl": round(float(thr.mean()), 2) if mine.any() else 0.0,
            "DRB.UEThpDl_p5": round(float(np.percentile(thr, 5)), 2) if mine.any() else 0.0,
            "PEE.AvgPower": round(float(PSLEEP_W if asleep[c] else P0_W + DELTA_P * PMAX_W * busy[c]), 1),
            "edge_share": round(float(near_border[mine].mean()), 3) if mine.any() else 0.0,
        })
    return reports


def summary(result):
    """A few network-wide numbers used to judge an action."""
    thr, demand = result["throughput_mbps"], result["demand_mbps"]
    busy = np.minimum(result["load"], 1.0)
    power = np.where(result["asleep"], PSLEEP_W, P0_W + DELTA_P * PMAX_W * busy)
    return {
        "worst_cell_util_pct": round(100 * float(busy.max()), 1),
        "users_fully_served_pct": round(100 * float((thr >= 0.999 * demand).mean()), 1),
        "poor_signal_users_pct": round(100 * float((result["sinr_db"] < POOR_SINR_DB).mean()), 1),
        "total_power_w": round(float(power.sum()), 0),
    }
