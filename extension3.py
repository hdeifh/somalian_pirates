#!/usr/bin/env python3
"""
ECE4191 Module 3 - Experiment 4 (Option 3)
Feeder-wide, per-node MPC with voltage compliance + bill minimisation,
closed-loop (CIL) over Modbus TCP.

Built directly on top of:
  - Experiment 2 (mpc.py)  : feeder-wide rolling MPC, CSV loaders, Modbus I/O
  - Experiment 3 (cil.py)  : Controller-in-the-Loop closed-loop SoC feedback

WHAT'S DIFFERENT FROM EXPERIMENT 2
-----------------------------------
Experiment 2 solves ONE aggregate battery variable per step and then
*disaggregates* it across the 16 node-phase groups by customer count.
That cannot respect per-node voltage limits, because it has no freedom to
put more/less battery power at any individual node.

This script instead gives every node-phase battery group its OWN decision
variable in the QP (shape (K, H) where K = 16 node-phase groups, H = horizon
steps), and adds a linearised voltage constraint at every node, every step:

    V_k(t) ~= V_BASE_PU + S_k * ( PPV_k(t) - Pload_k(t) + Pbat_k(t) )

S_k is the node's voltage sensitivity to real power injection (pu/kW).
THESE ARE PLACEHOLDER VALUES (see V_SENS_DEFAULT_PU_PER_KW below) --
replace them with sensitivities computed from the feeder's Ybus / power-flow
Jacobian (or from your Module 3 admittance matrix), or supply a CSV via
--v-sens-file with columns node,sens_pu_per_kw. Getting this right is a
core part of the assignment, not something this script can do for you
without your model's admittance data.

WHAT'S DIFFERENT FROM EXPERIMENT 3
-----------------------------------
Experiment 3 (cil.py) runs true closed-loop CIL, but only for Node 646 --
every other node is hardcoded to Pbat=0. This script generalises CIL
feedback to every node-phase group: SoC feedback is read from whichever
Modbus input registers you configure in SOC_INPUT_REGISTERS below (only
646_B is populated by default, matching the HIL model in cil.py); any
node without a configured register falls back to model-predicted SoC,
exactly like cil.py's fallback path. The same pattern is used for
optional measured voltage feedback via VOLTAGE_INPUT_REGISTERS, which is
empty by default -- populate it if your Typhoon model exposes per-node
voltage magnitude on input registers.

Objective (bill minimisation, Module 2 style, generalised across nodes):
    minimize sum_t [ -delta * eta(t) * sum_k(x1_k(t))
                      + w * eta(t) * (sum_k x2_k(t))^2 ]
    x2_k(t) = Pload_k(t) - PPV_k(t) - x1_k(t)     (per-node grid import)

The linear term rewards discharging into high-tariff periods (bill
savings); the quadratic term on TOTAL feeder import discourages large
swings/peaks, same as Experiment 2's objective, just summed over nodes.

Per-node forecasts are built by splitting the feeder-wide day-ahead
forecast (same CSV as Experiment 2) proportionally by customer count --
there is no per-node day-ahead forecast file, so this is the best
available approximation; PV is split the same way for lack of per-node
PV-capacity data. Actual playback data (for logging/CIL feedback) is
read per node-phase directly from the Experiment 2/3 actual CSV.

Test first with:
    python experiment4_voltage_bill_mpc.py --dry-run --no-wait --verbose

Live:
    python experiment4_voltage_bill_mpc.py --w 1.0 --no-prompt
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:
    print("ERROR: cvxpy is not installed.")
    print("Install with: pip install cvxpy pandas numpy pymodbus --break-system-packages")
    sys.exit(1)

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    print("ERROR: pymodbus is not installed.")
    print("Install with: pip install pymodbus --break-system-packages")
    sys.exit(1)


# ============================================================
# 1. Experiment 4 settings
# ============================================================

N_STEPS = 48
DELTA_HOURS = 0.5
HORIZON_STEPS = 48

PLAYBACK_DAYS = 4
PLAYBACK_STEPS = PLAYBACK_DAYS * N_STEPS

STEP_SECONDS = 2.0

TOTAL_CUSTOMERS = 1330

# Per-customer battery sizing (same convention as cil.py's Node 646 battery).
KWH_PER_CUSTOMER = 10.0
KW_PER_CUSTOMER = 5.0

P_UPPER_KW = 3000
P_LOWER_KW = -1500

DEFAULT_W = 1.0

# --- Voltage compliance -------------------------------------------------
# Feeder voltage limits as specified for this task, in volts (line-to-
# neutral). V_BASE_VOLTS is the nominal line-to-neutral voltage used to
# convert volts <-> per-unit for the linearised sensitivity model, which
# works in pu internally. Default base = 4160 V line-line / sqrt(3), the
# standard IEEE 13-Node Feeder main-line nominal voltage; override with
# --v-base-volts if your model uses a different nominal.
V_MIN_VOLTS_DEFAULT = 2285.0
V_MAX_VOLTS_DEFAULT = 2522.0
V_BASE_VOLTS_DEFAULT = 4160.0 / np.sqrt(3)  # ~2401.8 V
V_BASE_PU = 1.0  # linearisation point (always 1.0 pu regardless of volts base)

# PLACEHOLDER sensitivity: replace with real Ybus/Jacobian-derived values,
# or supply --v-sens-file (columns: node,sens_pu_per_kw). Sign convention:
# positive real power INJECTED at a node (PV or battery discharge) RAISES
# its voltage, so this should be entered as a positive number in pu/kW for
# a typical radial feeder; the code applies the sign internally.
V_SENS_DEFAULT_PU_PER_KW = 0.00020

# ------------------------------------------------------------
# Modbus
# ------------------------------------------------------------
DEFAULT_HIL_IP = "192.168.1.210"
DEFAULT_HIL_PORT = 502

HOLDING_START = 2000
HOLDING_COUNT = 64

SIGNED_16BIT_MIN = -32768
SIGNED_16BIT_MAX = 32767

RECONNECT_RETRIES = 5
RECONNECT_DELAY_S = 2.0

# Measured SoC feedback (input registers, %). Only nodes present here get
# true CIL feedback; everything else uses model-predicted SoC, same
# fallback behaviour as cil.py.
SOC_INPUT_REGISTERS: dict[str, int] = {
    "646_B": 3000,
}

# Measured voltage feedback (input registers, pu x100 or similar --
# ADJUST THE SCALING in read_voltage_pu() to match your HIL model).
# Empty by default: no node has one configured, so the QP relies purely
# on the linearised sensitivity model until you populate this.
VOLTAGE_INPUT_REGISTERS: dict[str, int] = {}


# ============================================================
# 2. 16 node-phase configuration
# ============================================================

NODES = {
    "646_B": {"register": 2000, "customers": 102, "order": "normal",   "qref": 132},
    "645_B": {"register": 2004, "customers": 63,  "order": "normal",   "qref": 125},
    "611_C": {"register": 2008, "customers": 68,  "order": "normal",   "qref": 80},
    "652_A": {"register": 2012, "customers": 46,  "order": "normal",   "qref": 86},

    "671_A": {"register": 2016, "customers": 159, "order": "normal",   "qref": 220},
    "671_B": {"register": 2020, "customers": 155, "order": "normal",   "qref": 220},
    "671_C": {"register": 2024, "customers": 159, "order": "normal",   "qref": 220},

    "692_C": {"register": 2028, "customers": 66,  "order": "reversed", "qref": 151},
    "692_B": {"register": 2032, "customers": 0,   "order": "reversed", "qref": 0},
    "692_A": {"register": 2036, "customers": 0,   "order": "reversed", "qref": 0},

    "675_C": {"register": 2040, "customers": 119, "order": "reversed", "qref": 212},
    "675_B": {"register": 2044, "customers": 36,  "order": "reversed", "qref": 60},
    "675_A": {"register": 2048, "customers": 191, "order": "reversed", "qref": 190},

    "634_C": {"register": 2052, "customers": 52,  "order": "reversed", "qref": 132},
    "634_B": {"register": 2056, "customers": 45,  "order": "reversed", "qref": 132},
    "634_A": {"register": 2060, "customers": 69,  "order": "reversed", "qref": 132},
}

assert sum(v["customers"] for v in NODES.values()) == TOTAL_CUSTOMERS

NODE_KEYS = list(NODES.keys())          # fixed row order for the QP matrix
K_NODES = len(NODE_KEYS)

_PHASE_LETTER_MAP = {"PH1": "A", "PH2": "B", "PH3": "C", "A": "A", "B": "B", "C": "C"}


def _phase_to_letter(value) -> str | None:
    return _PHASE_LETTER_MAP.get(str(value).strip().upper())


# ============================================================
# 3. Small utilities (unchanged from mpc.py)
# ============================================================

def _norm(s) -> str:
    return (
        str(s).strip().lower()
        .replace(" ", "").replace("_", "").replace("-", "")
        .replace(".", "").replace("/", "")
    )


def _action(v: float) -> str:
    if v > 0.5:
        return "Discharge"
    if v < -0.5:
        return "Charge"
    return "Idle"


def _numeric_series(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return arr[np.isfinite(arr)]


def _find_column(df: pd.DataFrame, aliases, required=True):
    normalised = {_norm(c): c for c in df.columns}
    for alias in aliases:
        a = _norm(alias)
        if a in normalised:
            return normalised[a]
    for c in df.columns:
        nc = _norm(c)
        for alias in aliases:
            a = _norm(alias)
            if a and (a in nc or nc in a):
                return c
    if required:
        raise ValueError(f"Could not find a column matching {aliases}. Available: {list(df.columns)}")
    return None


_DAY_BLOCK_METADATA_KEYS = {"date", "node", "phase", "ncustomers", "profile"}


def _time_of_day_columns(columns) -> list:
    cols = []
    for c in columns:
        if str(c).startswith("Unnamed"):
            continue
        if _norm(c) in _DAY_BLOCK_METADATA_KEYS:
            continue
        cols.append(c)
    return cols


def _is_day_blocked(df: pd.DataFrame, require_node: bool) -> bool:
    cols_norm = {_norm(c) for c in df.columns}
    if "date" not in cols_norm or "profile" not in cols_norm:
        return False
    if require_node and ("node" not in cols_norm or "phase" not in cols_norm):
        return False
    return len(_time_of_day_columns(df.columns)) >= 24


def _parse_node_from_value(value) -> str | None:
    s = str(value).strip().upper().replace("-", "_").replace(" ", "")
    if s in NODES:
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    phase = None
    for p in ("A", "B", "C"):
        if s.endswith(p) or f"_{p}" in s or f"PH{p}" in s:
            phase = p
            break
    if digits and phase:
        candidate = f"{digits}_{phase}"
        if candidate in NODES:
            return candidate
    return None


def _extract_node_from_column(col) -> str | None:
    s = str(col).upper().replace("-", "_").replace(" ", "")
    for node in NODES:
        if node in s:
            return node
    digits = "".join(ch for ch in s if ch.isdigit())
    phase = None
    if "PHA" in s or "_A" in s:
        phase = "A"
    elif "PHB" in s or "_B" in s:
        phase = "B"
    elif "PHC" in s or "_C" in s:
        phase = "C"
    if digits and phase:
        candidate = f"{digits}_{phase}"
        if candidate in NODES:
            return candidate
    return None


# ============================================================
# 4. Forecast loader (feeder-wide aggregate -- unchanged from mpc.py)
# ============================================================

def _load_forecast_day_blocked(df: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    date_col = _find_column(df, ["Date"])
    profile_col = _find_column(df, ["Profile"])
    time_cols = _time_of_day_columns(df.columns)

    work = df.copy()
    work[profile_col] = work[profile_col].astype(str).str.strip()
    work["_date_sort"] = pd.to_datetime(work[date_col], format="%d-%b-%y", errors="coerce")
    if work["_date_sort"].isna().any():
        work["_date_sort"] = pd.to_datetime(work[date_col], errors="coerce")
    if work["_date_sort"].isna().any():
        bad = sorted(set(work.loc[work["_date_sort"].isna(), date_col].astype(str)))
        raise ValueError(f"Could not parse Date value(s) {bad} in {path.name}.")

    dates = sorted(work["_date_sort"].unique())
    load_chunks, pv_chunks = [], []
    for d in dates:
        day = work[work["_date_sort"] == d]
        load_row = day[day[profile_col].str.contains("load", case=False)]
        pv_row = day[day[profile_col].str.contains("pv", case=False)]
        if len(load_row) != 1 or len(pv_row) != 1:
            raise ValueError(f"Expected exactly one load/PV row for {pd.Timestamp(d).date()} in {path.name}.")
        load_chunks.append(_numeric_series(load_row.iloc[0][time_cols]))
        pv_chunks.append(_numeric_series(pv_row.iloc[0][time_cols]))

    load = np.concatenate(load_chunks)
    pv = np.concatenate(pv_chunks)
    if len(load) != len(pv):
        raise ValueError(f"Forecast load/PV concatenated lengths differ in {path.name}.")
    return load, pv


def load_forecast(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if _is_day_blocked(df, require_node=False):
        return _load_forecast_day_blocked(df, path)

    load_col = _find_column(df, ["P_load", "P_load_kW", "Pload", "Pload_kW", "load",
                                  "load_kW", "load_forecast", "P_load_hat", "Pload_hat"], required=False)
    pv_col = _find_column(df, ["P_PV", "P_PV_kW", "PPV", "PPV_kW", "PV", "PV_kW",
                                "pv_generation", "P_PV_hat", "PPV_hat"], required=False)

    if load_col is not None and pv_col is not None:
        load = _numeric_series(df[load_col])
        pv = _numeric_series(df[pv_col])
        if len(load) != len(pv):
            raise ValueError("Forecast load and PV columns have different lengths.")
        return load, pv

    raw = pd.read_csv(path, header=None, index_col=0)
    raw.index = [str(i).strip() for i in raw.index]

    def find_row(needle):
        for idx in raw.index:
            if needle in str(idx).lower():
                return idx
        return None

    load_key = find_row("load")
    pv_key = find_row("pv")
    if load_key is None or pv_key is None:
        raise ValueError(
            "Could not identify feeder forecast load/PV. Expected a day-blocked CSV "
            "(Date/Profile/time columns), P_load/P_PV columns, or labelled rows."
        )
    load = _numeric_series(raw.loc[load_key])
    pv = _numeric_series(raw.loc[pv_key])
    if len(load) != len(pv):
        raise ValueError("Forecast load and PV lengths differ.")
    return load, pv


# ============================================================
# 5. Actual 16-node playback loader (unchanged from mpc.py)
# ============================================================

def _load_actual_day_blocked_nodes(df: pd.DataFrame, path: Path):
    date_col = _find_column(df, ["Date"])
    node_col = _find_column(df, ["Node"])
    phase_col = _find_column(df, ["Phase"])
    profile_col = _find_column(df, ["Profile"])
    time_cols = _time_of_day_columns(df.columns)

    work = df.copy()
    work[profile_col] = work[profile_col].astype(str).str.strip()
    work["_date_sort"] = pd.to_datetime(work[date_col], format="%d-%b-%y", errors="coerce")
    if work["_date_sort"].isna().any():
        work["_date_sort"] = pd.to_datetime(work[date_col], errors="coerce")
    if work["_date_sort"].isna().any():
        bad = sorted(set(work.loc[work["_date_sort"].isna(), date_col].astype(str)))
        raise ValueError(f"Could not parse Date value(s) {bad} in {path.name}.")

    dates = sorted(work["_date_sort"].unique())
    phase_letters = work[phase_col].map(_phase_to_letter)
    if phase_letters.isna().any():
        bad = sorted(set(work.loc[phase_letters.isna(), phase_col].astype(str)))
        raise ValueError(f"Unrecognised Phase value(s) {bad} in {path.name}.")
    work["_node_phase"] = work[node_col].astype(str).str.strip() + "_" + phase_letters

    actual = {}
    for node in NODES:
        node_rows = work[work["_node_phase"] == node]
        if node_rows.empty:
            raise ValueError(f"No rows found for node-phase '{node}' in {path.name}.")
        load_chunks, pv_chunks = [], []
        for d in dates:
            day = node_rows[node_rows["_date_sort"] == d]
            load_row = day[day[profile_col].str.contains("load", case=False)]
            pv_row = day[day[profile_col].str.contains("pv", case=False)]
            if len(load_row) != 1 or len(pv_row) != 1:
                raise ValueError(
                    f"Expected exactly one load/PV row for '{node}' on "
                    f"{pd.Timestamp(d).date()} in {path.name}."
                )
            load_chunks.append(_numeric_series(load_row.iloc[0][time_cols]))
            pv_chunks.append(_numeric_series(pv_row.iloc[0][time_cols]))
        actual[node] = {"load": np.concatenate(load_chunks), "pv": np.concatenate(pv_chunks)}

    lengths = {len(v["load"]) for v in actual.values()}
    if len(lengths) != 1:
        raise ValueError(f"Node-phase actual profiles have different lengths: {lengths}")
    return actual, next(iter(lengths))


def load_actual_node_profiles(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Actual CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if _is_day_blocked(df, require_node=True):
        return _load_actual_day_blocked_nodes(df, path)

    node_col = _find_column(df, ["NodePhase", "Node_Phase", "Node", "Bus", "NodeID"], required=False)
    phase_col = _find_column(df, ["Phase", "Ph"], required=False)
    load_col = _find_column(df, ["Pload", "P_load", "Pload_kW", "P_load_kW", "Load", "Load_kW"], required=False)
    pv_col = _find_column(df, ["PPV", "P_PV", "PPV_kW", "P_PV_kW", "PV", "PV_kW"], required=False)

    if node_col is not None and load_col is not None and pv_col is not None:
        if phase_col is not None:
            node_values = [_parse_node_from_value(f"{n}_{ph}") for n, ph in zip(df[node_col], df[phase_col])]
        else:
            node_values = [_parse_node_from_value(v) for v in df[node_col]]

        df2 = df.copy()
        df2["_node_phase_internal"] = node_values

        if df2["_node_phase_internal"].notna().sum() > 0:
            actual = {}
            for node in NODES:
                rows = df2[df2["_node_phase_internal"] == node]
                if len(rows) == 0:
                    raise ValueError(f"Actual CSV does not contain node-phase {node}.")
                actual[node] = {"load": _numeric_series(rows[load_col]), "pv": _numeric_series(rows[pv_col])}
            lengths = {len(v["load"]) for v in actual.values()}
            if len(lengths) != 1:
                raise ValueError(f"Node-phase actual profiles have different lengths: {lengths}")
            return actual, next(iter(lengths))

    actual = {node: {"load": None, "pv": None} for node in NODES}
    for col in df.columns:
        node = _extract_node_from_column(col)
        if node is None:
            continue
        nc = _norm(col)
        if any(key in nc for key in ["ppv", "ppvk", "pvpower", "p_pv"]):
            actual[node]["pv"] = _numeric_series(df[col])
        elif any(key in nc for key in ["pload", "ploadkw", "loadkw", "loadpower", "p_load"]):
            actual[node]["load"] = _numeric_series(df[col])

    if all(actual[node]["load"] is not None and actual[node]["pv"] is not None for node in NODES):
        lengths = {len(actual[node]["load"]) for node in NODES}
        if len(lengths) != 1:
            raise ValueError(f"Wide actual profiles have different lengths: {lengths}")
        return actual, next(iter(lengths))

    raw = pd.read_csv(path, header=None)
    fallback = {node: {"load": None, "pv": None} for node in NODES}
    for r in range(len(raw)):
        row_text = " ".join(str(x) for x in raw.iloc[r, :4].tolist()).upper()
        node = None
        for candidate in NODES:
            if candidate in row_text:
                node = candidate
                break
        if node is None:
            continue
        if "PV" in row_text:
            signal = "pv"
        elif "LOAD" in row_text or "PLOAD" in row_text:
            signal = "load"
        else:
            continue
        fallback[node][signal] = _numeric_series(raw.iloc[r, 4:])

    if all(fallback[node]["load"] is not None and fallback[node]["pv"] is not None for node in NODES):
        lengths = {len(v["load"]) for v in fallback.values()}
        if len(lengths) != 1:
            raise ValueError(f"Fallback actual profiles have different lengths: {lengths}")
        return fallback, next(iter(lengths))

    raise ValueError(
        "Could not identify all 16 node-phase load/PV profiles in the actual CSV.\n"
        f"Columns found: {list(df.columns)}"
    )


# ============================================================
# 6. Per-node forecast (proportional split of the aggregate forecast)
# ============================================================

def build_node_forecast(forecast_load: np.ndarray, forecast_pv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the feeder-wide day-ahead forecast into a (K, T) array per node,
    proportional to customer count. No per-node day-ahead forecast file
    exists, so this is the best available approximation for both load and
    PV (see module docstring).
    """
    shares = np.array([NODES[n]["customers"] / TOTAL_CUSTOMERS for n in NODE_KEYS])  # (K,)
    load_node = shares[:, None] * forecast_load[None, :]   # (K, T)
    pv_node = shares[:, None] * forecast_pv[None, :]        # (K, T)
    return load_node, pv_node


def load_voltage_sensitivities(path: Path | None) -> np.ndarray:
    """
    Returns an array of length K (pu/kW) in NODE_KEYS order.
    Falls back to V_SENS_DEFAULT_PU_PER_KW for any node not in the file,
    or for every node if no file is given.
    """
    sens = np.full(K_NODES, V_SENS_DEFAULT_PU_PER_KW, dtype=float)
    if path is None:
        print(f"  Using placeholder voltage sensitivity {V_SENS_DEFAULT_PU_PER_KW} pu/kW for all nodes.")
        return sens

    if not path.exists():
        raise FileNotFoundError(f"Voltage sensitivity CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    node_col = _find_column(df, ["node", "node_phase", "nodephase"])
    sens_col = _find_column(df, ["sens_pu_per_kw", "sensitivity", "dv_dp", "sens"])

    found = 0
    for _, row in df.iterrows():
        key = _parse_node_from_value(row[node_col]) or str(row[node_col]).strip()
        if key in NODE_KEYS:
            sens[NODE_KEYS.index(key)] = float(row[sens_col])
            found += 1

    print(f"  Loaded voltage sensitivities for {found}/{K_NODES} nodes from {path.name}; "
          f"remaining nodes use the {V_SENS_DEFAULT_PU_PER_KW} pu/kW placeholder.")
    return sens


# ============================================================
# 7. Module-2 TOU tariff (unchanged)
# ============================================================

def generate_eta_profiles(total_steps):
    eta_day = np.zeros(N_STEPS)
    eta_day[np.r_[0:14, 44:48]] = 0.03
    eta_day[np.r_[14:28, 40:44]] = 0.06
    eta_day[28:40] = 0.30
    n_days = int(np.ceil(total_steps / N_STEPS))
    return np.tile(eta_day, n_days)[:total_steps]


# ============================================================
# 8. Feeder-wide, per-node MPC QP with voltage constraints
# ============================================================

def solve_mpc_voltage_horizon(
    load_win_node,   # (K, H)
    pv_win_node,      # (K, H)
    soc0_node,        # (K,)
    capacity_kwh_node,  # (K,)
    batt_lim_node,     # (K,)
    v_sens_node,       # (K,) pu/kW
    v_min_pu,
    v_max_pu,
    p_upper_kw,
    p_lower_kw,
    solver,
    eta_pred,          # (H,)
    w,
    enforce_terminal_soc,
):
    """
    x1[k,t] = battery power at node k, step t (+ve = discharge)
    x2[k,t] = grid import at node k, step t
    z[k,t]  = stored energy at node k, step t

    Objective mirrors Experiment 2's Module-2 objective, generalised to
    sum battery discharge and TOTAL feeder grid power across all nodes:

        min sum_t [ -delta*eta(t)*sum_k(x1[k,t])
                     + w*eta(t)*(sum_k x2[k,t])^2 ]

    plus linearised per-node voltage compliance constraints.
    """
    K, H = load_win_node.shape
    eta_pred = np.asarray(eta_pred, dtype=float).reshape(-1)
    if len(eta_pred) != H:
        raise ValueError("eta_pred length does not match MPC horizon.")
    if w <= 0:
        raise ValueError("w must be > 0.")

    x1 = cp.Variable((K, H))
    x2 = cp.Variable((K, H))

    z = cp.Variable((K, H))
    soc0_col = soc0_node.reshape(K, 1)
    z_expr = soc0_col - cp.cumsum(DELTA_HOURS * x1, axis=1)

    constraints = [
        x2 == load_win_node - pv_win_node - x1,
        z == z_expr,

        x1 <= batt_lim_node.reshape(K, 1),
        x1 >= -batt_lim_node.reshape(K, 1),

        z >= 0.0,
        z <= capacity_kwh_node.reshape(K, 1),
    ]

    if enforce_terminal_soc:
        constraints.append(z[:, -1] == soc0_node)

    x2_total = cp.sum(x2, axis=0)  # (H,)
    constraints += [x2_total <= p_upper_kw, x2_total >= p_lower_kw]

    # --- Linearised voltage compliance (per node, per step) -------------
    # V[k,t] = V_BASE_PU + v_sens[k] * ( PV[k,t] - Pload[k,t] + x1[k,t] )
    p_inj_forecast = pv_win_node - load_win_node   # (K, H), constant part
    v_sens_col = v_sens_node.reshape(K, 1)
    v_expr = V_BASE_PU + cp.multiply(v_sens_col, p_inj_forecast + x1)
    constraints += [v_expr >= v_min_pu, v_expr <= v_max_pu]

    x1_total = cp.sum(x1, axis=0)  # (H,)
    objective = cp.Minimize(
        cp.sum(-DELTA_HOURS * cp.multiply(eta_pred, x1_total)
               + w * cp.multiply(eta_pred, cp.square(x2_total)))
    )

    problem = cp.Problem(objective, constraints)
    solver_name = solver.upper()
    if not hasattr(cp, solver_name):
        raise ValueError(f"CVXPY solver {solver_name} is not available.")

    problem.solve(solver=getattr(cp, solver_name), verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MPC QP failed. Solver status = {problem.status}")

    return (
        np.asarray(x1.value, dtype=float),   # (K, H)
        np.asarray(x2.value, dtype=float),   # (K, H)
        np.asarray(z.value, dtype=float),    # (K, H)
        str(problem.status),
        float(problem.value),
    )


# ============================================================
# 9. Modbus (unchanged from mpc.py)
# ============================================================

class ModbusConnection:
    def __init__(self, ip: str, port: int, retries: int = RECONNECT_RETRIES, delay: float = RECONNECT_DELAY_S):
        self.ip, self.port, self.retries, self.delay = ip, port, retries, delay
        self.client = None

    def connect(self):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                client = ModbusTcpClient(self.ip, port=self.port)
                if client.connect():
                    self.client = client
                    return
            except Exception as exc:
                last_exc = exc
            print(f"Connection attempt {attempt}/{self.retries} failed"
                  f"{f' ({last_exc})' if last_exc else ''}; retrying in {self.delay}s...")
            time.sleep(self.delay)
        raise ConnectionError(f"Cannot connect to HIL Modbus server {self.ip}:{self.port}")

    def reconnect(self):
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass
        self.client = None
        self.connect()

    def close(self):
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass


def signed_to_register(value: float) -> int:
    value = int(round(float(value)))
    if value < SIGNED_16BIT_MIN or value > SIGNED_16BIT_MAX:
        clamped = max(SIGNED_16BIT_MIN, min(SIGNED_16BIT_MAX, value))
        print(f"WARNING: {value} outside signed 16-bit range; clamped to {clamped}")
        value = clamped
    return value & 0xFFFF


def build_node_payload(pref_kw, qref_kvar, pv_kw, batt_kw, order):
    if order == "normal":
        values = [pref_kw, qref_kvar, pv_kw, batt_kw]
    elif order == "reversed":
        values = [batt_kw, pv_kw, qref_kvar, pref_kw]
    else:
        raise ValueError(f"Unknown register order: {order}")
    return [signed_to_register(v) for v in values]


def modbus_write_node(conn, register_start, payload, dry_run, verbose=False):
    if dry_run:
        if verbose:
            print(f"    [DRY-RUN] holding {register_start}-{register_start+3}: {payload}")
        return
    try:
        result = conn.client.write_registers(address=register_start, values=payload)
        if result is None or result.isError():
            raise IOError("write_registers returned an error")
        return
    except Exception as exc:
        print(f"WARNING: Modbus write failed: {exc}")
    try:
        print("  Reconnecting and retrying write...")
        conn.reconnect()
        result = conn.client.write_registers(address=register_start, values=payload)
        if result is None or result.isError():
            print("ERROR: write failed after reconnection.")
    except Exception as exc:
        print(f"ERROR: reconnect/write failed: {exc}")


def clear_all_registers(conn, dry_run):
    payload = [0] * HOLDING_COUNT
    if dry_run:
        print(f"[DRY-RUN] clear holding {HOLDING_START}-{HOLDING_START+HOLDING_COUNT-1}")
        return
    result = conn.client.write_registers(address=HOLDING_START, values=payload)
    if result is None or result.isError():
        print("WARNING: could not clear holding registers.")
    else:
        print("Holding registers cleared.")


# ============================================================
# 10. CIL feedback: per-node SoC and (optional) voltage
# ============================================================

def read_soc_percent(conn, register: int) -> float | None:
    """Generalised version of cil.py's SoC reader -- any node's register."""
    if conn is None or conn.client is None:
        return None
    try:
        result = conn.client.read_input_registers(address=register, count=1)
        if result is None or result.isError():
            return None
        return float(result.registers[0]) / 100.0
    except Exception:
        return None


def read_voltage_pu(conn, register: int) -> float | None:
    """
    Reads a per-node voltage-magnitude input register.
    ADJUST THE SCALING FACTOR below to match how your Typhoon model
    encodes voltage on this register (e.g. pu*1000, or volts).
    Returns None (falls back to the linearised model) if unavailable.
    """
    if conn is None or conn.client is None:
        return None
    try:
        result = conn.client.read_input_registers(address=register, count=1)
        if result is None or result.isError():
            return None
        return float(result.registers[0]) / 1000.0  # <-- adjust scaling to your model
    except Exception:
        return None


# ============================================================
# 11. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ECE4191 Module 3 Experiment 4: feeder-wide per-node MPC "
            "with voltage compliance + bill minimisation, closed-loop CIL"
        )
    )
    parser.add_argument("--forecast", default="central_agg_forecast_data_students.csv",
                        help="Feeder-wide day-ahead forecast CSV.")
    parser.add_argument("--actual", default="agg_jan2013_students.csv",
                        help="Actual 16-node playback CSV.")
    parser.add_argument("--v-sens-file", default=None,
                        help="Optional CSV (node,sens_pu_per_kw) of real voltage sensitivities. "
                             "Without this, ALL nodes use the placeholder value in the script header.")
    parser.add_argument("--v-min-volts", type=float, default=V_MIN_VOLTS_DEFAULT,
                        help=f"Lower voltage limit in volts (line-to-neutral). Default {V_MIN_VOLTS_DEFAULT}.")
    parser.add_argument("--v-max-volts", type=float, default=V_MAX_VOLTS_DEFAULT,
                        help=f"Upper voltage limit in volts (line-to-neutral). Default {V_MAX_VOLTS_DEFAULT}.")
    parser.add_argument("--v-base-volts", type=float, default=V_BASE_VOLTS_DEFAULT,
                        help=f"Nominal line-to-neutral voltage used to convert volts<->pu. "
                             f"Default {V_BASE_VOLTS_DEFAULT:.2f} (4160V line-line / sqrt3).")
    parser.add_argument("--w", type=float, default=DEFAULT_W,
                        help="Module-2 quadratic grid-power weight.")
    parser.add_argument("--horizon-steps", type=int, default=HORIZON_STEPS)
    parser.add_argument("--playback-steps", type=int, default=PLAYBACK_STEPS)
    parser.add_argument("--step-seconds", type=float, default=STEP_SECONDS)
    parser.add_argument("--p-upper", type=float, default=P_UPPER_KW)
    parser.add_argument("--p-lower", type=float, default=P_LOWER_KW)
    parser.add_argument("--terminal-soc", action="store_true",
                        help="Enforce z[k,H]==z[k,0] per node (off by default, D-RHO style as in cil.py).")
    parser.add_argument("--solver", default="OSQP")
    parser.add_argument("--ip", default=DEFAULT_HIL_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_HIL_PORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep-final", action="store_true")
    parser.add_argument("--no-cil", action="store_true",
                        help="Disable measured SoC/voltage feedback; run open-loop (model state only).")
    args = parser.parse_args()

    if args.horizon_steps != 48:
        print("WARNING: this experiment specifies a 48-step 24-hour MPC horizon.")
    if args.playback_steps != 192:
        print("WARNING: this experiment specifies 4 days = 192 half-hour playback steps.")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    forecast_path = Path(args.forecast)
    actual_path = Path(args.actual)

    print("\nLoading feeder-wide forecast...")
    forecast_load, forecast_pv = load_forecast(forecast_path)
    print(f"  Forecast samples: {len(forecast_load)}")

    print("\nLoading actual 16-node profiles...")
    actual, actual_steps = load_actual_node_profiles(actual_path)
    print(f"  Actual samples/node-phase: {actual_steps}")

    if actual_steps < args.playback_steps:
        raise ValueError(f"Actual data contains only {actual_steps} steps; {args.playback_steps} required.")

    required_forecast_steps = args.playback_steps + args.horizon_steps - 1
    if len(forecast_load) < required_forecast_steps:
        raise ValueError(
            f"Forecast contains {len(forecast_load)} samples, but {required_forecast_steps} are needed."
        )

    print("\nBuilding per-node forecasts (proportional split by customer count)...")
    forecast_load_node, forecast_pv_node = build_node_forecast(forecast_load, forecast_pv)  # (K, T)

    print("\nLoading voltage sensitivities...")
    v_sens_node = load_voltage_sensitivities(Path(args.v_sens_file) if args.v_sens_file else None)

    eta = generate_eta_profiles(args.playback_steps + args.horizon_steps - 1)

    capacity_kwh_node = np.array([NODES[n]["customers"] * KWH_PER_CUSTOMER for n in NODE_KEYS])
    batt_lim_node = np.array([NODES[n]["customers"] * KW_PER_CUSTOMER for n in NODE_KEYS])
    soc_model = 0.5 * capacity_kwh_node.copy()  # 50% initial SoC per node

    # --------------------------------------------------------
    # Modbus connection
    # --------------------------------------------------------
    conn = None
    if not args.dry_run:
        print(f"\nConnecting to Typhoon HIL at {args.ip}:{args.port}...")
        conn = ModbusConnection(args.ip, args.port)
        conn.connect()
        print("  Connected.")

    # --------------------------------------------------------
    # Banner
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("ECE4191 MODULE 3 - EXPERIMENT 4")
    print("FEEDER-WIDE PER-NODE MPC: VOLTAGE COMPLIANCE + BILL MINIMISATION")
    print("=" * 78)
    print(f"  Node-phase groups   : {K_NODES}")
    print(f"  Customers           : {TOTAL_CUSTOMERS}")
    print(f"  Per-node battery    : {KWH_PER_CUSTOMER} kWh/cust, {KW_PER_CUSTOMER} kW/cust")
    print(f"  MPC horizon         : {args.horizon_steps} steps / 24 h")
    print(f"  Playback            : {args.playback_steps} steps / 4 days")
    print(f"  Objective weight    : w = {args.w}")
    print(f"  Voltage band        : [{args.v_min_pu}, {args.v_max_pu}] pu")
    print(f"  Voltage model       : linearised sensitivity "
          f"({'from ' + args.v_sens_file if args.v_sens_file else 'PLACEHOLDER, see script header'})")
    print(f"  Terminal SoC        : {'enforced per node' if args.terminal_soc else 'not enforced (D-RHO)'}")
    print(f"  CIL feedback        : {'disabled (open-loop)' if args.no_cil else 'enabled where registers configured'}")
    print(f"  SoC registers known : {list(SOC_INPUT_REGISTERS.keys())}")
    print(f"  Voltage registers   : {list(VOLTAGE_INPUT_REGISTERS.keys()) or '(none configured)'}")
    print(f"  Modbus registers    : 2000-2063")
    print("=" * 78)

    if args.dry_run:
        print("  * DRY-RUN: NO MODBUS WRITES *")

    if not args.no_prompt and not args.dry_run:
        print("\nPre-run checklist:")
        print("  1. MIEEE-13NF is compiled and running.")
        print("  2. SCADA Control Type = REMOTE CONTROL.")
        print("  3. BusSplitMap registers 2000-2063 are enabled.")
        print("  4. All 16 node-phase battery groups are enabled.")
        print("  5. Every node battery starts at 50% SoC.")
        print("  6. v_sens values reflect your feeder's actual sensitivities, not the placeholder.")
        input("\nPress Enter to start Experiment 4...")

    print("\nClearing holding registers...")
    clear_all_registers(conn, args.dry_run)

    if not args.no_wait and not args.dry_run:
        print("Waiting 12 s for HIL to settle...")
        time.sleep(12)

    # --------------------------------------------------------
    # Rolling MPC playback
    # --------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    aborted = False
    voltage_min_seen = np.inf
    voltage_max_seen = -np.inf

    print("\nStarting rolling per-node MPC playback...\n")

    try:
        for t in range(args.playback_steps):
            wall_start = time.time()

            start, end = t, t + args.horizon_steps
            load_win = forecast_load_node[:, start:end]  # (K, H)
            pv_win = forecast_pv_node[:, start:end]
            eta_win = eta[start:end]

            x1_seq, x2_seq, z_seq, status, obj_value = solve_mpc_voltage_horizon(
                load_win_node=load_win,
                pv_win_node=pv_win,
                soc0_node=soc_model,
                capacity_kwh_node=capacity_kwh_node,
                batt_lim_node=batt_lim_node,
                v_sens_node=v_sens_node,
                v_min_pu=args.v_min_pu,
                v_max_pu=args.v_max_pu,
                p_upper_kw=args.p_upper,
                p_lower_kw=args.p_lower,
                solver=args.solver,
                eta_pred=eta_win,
                w=args.w,
                enforce_terminal_soc=args.terminal_soc,
            )

            # Apply ONLY the first MPC action, per node.
            batt_first = x1_seq[:, 0]  # (K,)

            actual_load_total = 0.0
            actual_pv_total = 0.0
            predicted_v_first = (
                V_BASE_PU + v_sens_node * (pv_win[:, 0] - load_win[:, 0] + batt_first)
            )
            voltage_min_seen = min(voltage_min_seen, float(predicted_v_first.min()))
            voltage_max_seen = max(voltage_max_seen, float(predicted_v_first.max()))

            for i, node in enumerate(NODE_KEYS):
                p_load_node = float(actual[node]["load"][t])
                p_pv_node = float(actual[node]["pv"][t])
                actual_load_total += p_load_node
                actual_pv_total += p_pv_node

                batt_node = float(batt_first[i])

                payload = build_node_payload(
                    pref_kw=p_load_node,
                    qref_kvar=NODES[node]["qref"],
                    pv_kw=p_pv_node,
                    batt_kw=batt_node,
                    order=NODES[node]["order"],
                )
                modbus_write_node(conn, NODES[node]["register"], payload, args.dry_run, args.verbose)

            baseline_grid = actual_load_total - actual_pv_total
            actual_grid = baseline_grid - float(batt_first.sum())

            # --------------------------------------------------
            # Advance model SoC using the applied action only
            # --------------------------------------------------
            soc_before = soc_model.copy()
            soc_model = np.clip(soc_model - DELTA_HOURS * batt_first, 0.0, capacity_kwh_node)

            # --------------------------------------------------
            # CIL feedback: overwrite model SoC wherever a register
            # is configured and reachable; identical fallback logic
            # to cil.py's SocLogger.
            # --------------------------------------------------
            soc_measured = {}
            if not args.no_cil and not args.dry_run:
                for node, reg in SOC_INPUT_REGISTERS.items():
                    pct = read_soc_percent(conn, reg)
                    if pct is not None:
                        idx = NODE_KEYS.index(node)
                        soc_model[idx] = np.clip(pct / 100.0 * capacity_kwh_node[idx], 0.0, capacity_kwh_node[idx])
                        soc_measured[node] = pct

            voltage_measured = {}
            if not args.no_cil and not args.dry_run and VOLTAGE_INPUT_REGISTERS:
                for node, reg in VOLTAGE_INPUT_REGISTERS.items():
                    v_pu = read_voltage_pu(conn, reg)
                    if v_pu is not None:
                        voltage_measured[node] = v_pu
                        voltage_min_seen = min(voltage_min_seen, v_pu)
                        voltage_max_seen = max(voltage_max_seen, v_pu)
                        if v_pu < args.v_min_pu or v_pu > args.v_max_pu:
                            print(f"  WARNING: measured voltage at {node} = {v_pu:.4f} pu "
                                  f"is outside [{args.v_min_pu}, {args.v_max_pu}]")

            # --------------------------------------------------
            # Log
            # --------------------------------------------------
            row = {
                "step": t + 1,
                "time_hours": round(t * DELTA_HOURS, 4),
                "Pload_actual_feeder_kw": round(actual_load_total, 4),
                "PPV_actual_feeder_kw": round(actual_pv_total, 4),
                "baseline_grid_kw": round(baseline_grid, 4),
                "Pbat_total_kw": round(float(batt_first.sum()), 4),
                "X2_grid_actual_kw": round(actual_grid, 4),
                "battery_action_total": _action(float(batt_first.sum())),
                "solver_status": status,
                "objective_value": round(obj_value, 8),
                "v_pu_min_predicted": round(float(predicted_v_first.min()), 5),
                "v_pu_max_predicted": round(float(predicted_v_first.max()), 5),
            }
            for i, node in enumerate(NODE_KEYS):
                row[f"{node}_Pbat_kw"] = round(float(batt_first[i]), 6)
                row[f"{node}_soc_pct"] = round(100.0 * soc_model[i] / capacity_kwh_node[i], 4) \
                    if capacity_kwh_node[i] > 0 else ""
                row[f"{node}_v_pu_predicted"] = round(float(predicted_v_first[i]), 5)
                if node in soc_measured:
                    row[f"{node}_soc_measured_pct"] = round(soc_measured[node], 4)
                if node in voltage_measured:
                    row[f"{node}_v_pu_measured"] = round(voltage_measured[node], 5)
            rows.append(row)

            if args.verbose or t == 0 or t == args.playback_steps - 1 or (t + 1) % 10 == 0:
                print(
                    f"Step {t+1:3d}/{args.playback_steps}: "
                    f"Load={actual_load_total:8.1f} kW | PV={actual_pv_total:8.1f} kW | "
                    f"Batt={batt_first.sum():8.1f} kW | Grid={actual_grid:8.1f} kW | "
                    f"V[min,max]=[{predicted_v_first.min():.4f},{predicted_v_first.max():.4f}] pu"
                )

            if not args.no_wait and not args.dry_run:
                elapsed = time.time() - wall_start
                time.sleep(max(0.0, args.step_seconds - elapsed))

    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
        aborted = True

    finally:
        if args.keep_final:
            print("\nFinal HIL values kept.")
        else:
            print("\nClearing holding registers...")
            clear_all_registers(conn, args.dry_run)
        if conn is not None:
            conn.close()

    # --------------------------------------------------------
    # Save + summary
    # --------------------------------------------------------
    output = Path(f"experiment4_voltage_bill_schedule_{ts}.csv")
    if rows:
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"\nOutput CSV saved to:\n  {output.resolve()}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 78)
        print("EXPERIMENT 4 COMPLETE")
        print("=" * 78)
        print(f"  Steps completed        : {len(df)}")
        print(f"  Non-optimal solves     : {sum(df['solver_status'] != 'optimal')}")
        print(f"  Baseline grid          : {df['baseline_grid_kw'].min():.1f} to {df['baseline_grid_kw'].max():.1f} kW")
        print(f"  MPC actual grid        : {df['X2_grid_actual_kw'].min():.1f} to {df['X2_grid_actual_kw'].max():.1f} kW")
        print(f"  Predicted voltage range: {voltage_min_seen:.4f} to {voltage_max_seen:.4f} pu "
              f"(limits [{args.v_min_pu}, {args.v_max_pu}])")
        if voltage_min_seen < args.v_min_pu or voltage_max_seen > args.v_max_pu:
            print("  *** VOLTAGE LIMIT VIOLATED at some step -- check v_sens accuracy / tighten margins ***")
        print("=" * 78)

    print("\nExperiment 4 finished." + (" (aborted)" if aborted else ""))


if __name__ == "__main__":
    main()