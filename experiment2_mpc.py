#!/usr/bin/env python3
"""
ECE4191 Module 3 - Experiment 2
Feeder-wide MPC + 16-node Modbus TCP playback

This program implements the feeder-wide MPC described in Experiment 2.

At every 30-minute step:
    1. Take a 48-step (24 h) day-ahead forecast window.
    2. Solve the Module-2 MPC QP using:
           min sum_k[-delta*eta_pred(k)*x1(k)
                     + w*eta_pred(k)*x2(k)^2]
           x2 = P_load_hat - P_PV_hat - x1
           z(k) = z0 - delta*x1(k)
           0 <= z(k) <= C
           -B <= x1(k) <= B
           z(H) = z0
    3. Apply ONLY x1(0), the first MPC battery action.
    4. Disaggregate the aggregate battery command across the 16
       node-phase battery groups according to customer count.
    5. Write actual Pload, Qload, PPV and the node-specific Pbat to
       Modbus holding registers 2000-2063.
    6. Advance one 30-minute step and solve again.

The feeder-wide controller uses the aggregate day-ahead forecast from:
    central_agg_forecast_data_students_formatted.csv

The real-time node-phase playback uses:
    agg_jan2013_students_transformed.csv

The CSV loader supports two shapes for both files:
  1) "Day-blocked" long format (matches the raw source data):
         Forecast : Date, N_Customers, Profile, <48 time-of-day columns>
         Actual   : Date, Node, Phase, N_Customers, Profile, <48 time-of-day columns>
     with one 'GC_Load_kW' row and one 'PV_Generation_kW' row per
     date (and per node-phase for the actual file). Multiple dates are
     concatenated in chronological order into one continuous time series.
     'Phase' may be given as Ph1/Ph2/Ph3 or directly as A/B/C.
  2) Flexible tabular / two-row-wide variants (kept for compatibility
     with other file layouts), matched via flexible column-name aliases.

IMPORTANT:
    eta_pred and w must be the same values used in your Module 2 MPC.
    Do not silently replace them with a different tariff or weight.

Battery sign convention:
    Pbat > 0 : discharge
    Pbat < 0 : charge

Modbus:
    2000-2003 : 646_B  [Pload, Qload, PPV, Pbat]
    ...
    2028-2039 : 692_C/B/A [Pbat, PPV, Qload, Pload]
    2040-2063 : 675_C/B/A and 634_C/B/A [Pbat, PPV, Qload, Pload]
"""
from __future__ import annotations
import argparse, sys, time, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

try: import cvxpy as cp
except ImportError: sys.exit("ERROR: cvxpy is not installed. Install with: pip install cvxpy pandas numpy pymodbus")
try: from pymodbus.client import ModbusTcpClient
except ImportError: sys.exit("ERROR: pymodbus is not installed. Install with: pip install pymodbus")

NUM_INTERVALS, HR_STEP, PREDICTION_STEPS = 48, 0.5, 48
SIM_DAYS, SIM_STEPS, SEC_PER_STEP = 4, 4 * 48, 2.0
MAX_CUSTOMERS, TOT_CAPACITY_KWH, TOT_POWER_KW = 1330, 10.0 * 1330, 5.0 * 1330
START_SOC_KWH, UPPER_P_KW, LOWER_P_KW, STD_W = 0.5 * 10.0 * 1330, 3000, -1500, 1.0
DEF_HIL_IP, DEF_HIL_PORT, REG_BEGIN, REG_NUM, SOC_INP_REG = "192.168.1.210", 502, 2000, 64, 3000
MIN_SIGNED_16BIT, MAX_SIGNED_16BIT, CONN_RETRIES, CONN_DELAY_S = -32768, 32767, 5, 2.0

NODE_CONFIG = {
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

_norm = lambda s: str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "").replace("/", "")
_action = lambda v: "Discharge" if v > 0.5 else ("Charge" if v < -0.5 else "Idle")
_numeric_series = lambda arr: pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(float)
_time_cols = lambda cols: [c for c in cols if not str(c).startswith("Unnamed") and _norm(c) not in {"date", "node", "phase", "ncustomers", "profile"}]

'''
AI Declaration: _extract_node, _find_col, load_forecast, load_actual_node_profiles were written with AI assistance
ModBusClient and SocReader classes were written by ECE4191 teaching staff (re-used from other toy_qp and toy_mpc files)
'''

def _extract_node(val):
    s, d = str(val).strip().upper().replace("-", "_").replace(" ", ""), "".join(filter(str.isdigit, str(val)))
    for n in NODE_CONFIG:
        if n in s: return n
    p = next((c for c in ("A", "B", "C") if s.endswith(c) or f"_{c}" in s or f"PH{c}" in s), None)
    return f"{d}_{p}" if d and p and f"{d}_{p}" in NODE_CONFIG else None

def _find_col(df, aliases, req=True):
    norm_cols = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in norm_cols: return norm_cols[_norm(a)]
    for c in df.columns:
        if any(_norm(a) in _norm(c) or _norm(c) in _norm(a) for a in aliases if _norm(a)): return c
    if req: raise ValueError(f"Missing column matching {aliases}.")
    return None

def load_forecast(filepath: Path):
    if not filepath.exists(): raise FileNotFoundError(f"Forecast CSV not found: {filepath}")
    df = pd.read_csv(filepath); df.columns = [str(c).strip() for c in df.columns]
    
    if "date" in {_norm(c) for c in df.columns} and "profile" in {_norm(c) for c in df.columns} and len(_time_cols(df.columns)) >= 24:
        c_dt, c_prof, t_cols = _find_col(df, ["Date"]), _find_col(df, ["Profile"]), _time_cols(df.columns)
        df["_dt"] = pd.to_datetime(df[c_dt], format="%d-%b-%y", errors="coerce").fillna(pd.to_datetime(df[c_dt], errors="coerce"))
        arr_ld, arr_pv = [], []
        for d in sorted(df["_dt"].dropna().unique()):
            day = df[df["_dt"] == d]
            arr_ld.append(_numeric_series(day[day[c_prof].str.contains("load", case=False, na=False)].iloc[0][t_cols]))
            arr_pv.append(_numeric_series(day[day[c_prof].str.contains("pv", case=False, na=False)].iloc[0][t_cols]))
        return np.concatenate(arr_ld), np.concatenate(arr_pv)
    
    cl, cpv = _find_col(df, ["P_load", "load", "P_load_hat"], False), _find_col(df, ["P_PV", "PV", "P_PV_hat"], False)
    if cl and cpv: return _numeric_series(df[cl]), _numeric_series(df[cpv])
    
    raw = pd.read_csv(filepath, header=None, index_col=0)
    return _numeric_series(raw.loc[next((i for i in raw.index if "load" in str(i).lower()), None)]), _numeric_series(raw.loc[next((i for i in raw.index if "pv" in str(i).lower()), None)])

def load_actual_node_profiles(filepath: Path):
    if not filepath.exists(): raise FileNotFoundError(f"Actual CSV not found: {filepath}")
    df = pd.read_csv(filepath); df.columns = [str(c).strip() for c in df.columns]
    
    if {"date", "node", "phase", "profile"}.issubset({_norm(c) for c in df.columns}):
        c_dt, c_nd, c_ph, c_pr, t_cols = _find_col(df, ["Date"]), _find_col(df, ["Node"]), _find_col(df, ["Phase"]), _find_col(df, ["Profile"]), _time_cols(df.columns)
        df["_dt"] = pd.to_datetime(df[c_dt], format="%d-%b-%y", errors="coerce").fillna(pd.to_datetime(df[c_dt], errors="coerce"))
        df["_np"] = df[c_nd].astype(str).str.strip() + "_" + df[c_ph].map(lambda v: {"PH1":"A","PH2":"B","PH3":"C","A":"A","B":"B","C":"C"}.get(str(v).strip().upper()))
        act = {}
        for n in NODE_CONFIG:
            nr = df[df["_np"] == n]
            act[n] = {"load": np.concatenate([_numeric_series(nr[(nr["_dt"] == d) & nr[c_pr].str.contains("load", case=False, na=False)].iloc[0][t_cols]) for d in sorted(df["_dt"].dropna().unique())]),
                      "pv": np.concatenate([_numeric_series(nr[(nr["_dt"] == d) & nr[c_pr].str.contains("pv", case=False, na=False)].iloc[0][t_cols]) for d in sorted(df["_dt"].dropna().unique())])}
        return act, len(next(iter(act.values()))["load"])
    
    cn, cl, cpv = _find_col(df, ["NodePhase", "Node"], False), _find_col(df, ["Pload", "Load"], False), _find_col(df, ["PPV", "PV"], False)
    if cn and cl and cpv:
        df["_np"] = df[cn].map(_extract_node) if not _find_col(df, ["Phase"], False) else [ _extract_node(f"{n}_{p}") for n, p in zip(df[cn], df[_find_col(df, ["Phase"])]) ]
        act = {n: {"load": _numeric_series(df[df["_np"] == n][cl]), "pv": _numeric_series(df[df["_np"] == n][cpv])} for n in NODE_CONFIG}
        return act, len(next(iter(act.values()))["load"])
    
    raw = pd.read_csv(filepath, header=None)
    fb = {n: {"load": None, "pv": None} for n in NODE_CONFIG}
    for r in range(len(raw)):
        txt, sig = " ".join(str(x) for x in raw.iloc[r, :4]).upper(), "pv" if "PV" in " ".join(str(x) for x in raw.iloc[r, :4]).upper() else ("load" if "LOAD" in " ".join(str(x) for x in raw.iloc[r, :4]).upper() else None)
        n = next((c for c in NODE_CONFIG if c in txt), None)
        if n and sig: fb[n][sig] = _numeric_series(raw.iloc[r, 4:])
    return fb, len(next(iter(fb.values()))["load"])

def generate_eta_profiles(total_intervals):
    eta = np.zeros(NUM_INTERVALS)
    eta[np.r_[0:14, 44:48]], eta[np.r_[14:28, 40:44]], eta[28:40] = 0.03, 0.06, 0.30
    return np.tile(eta, int(np.ceil(total_intervals / NUM_INTERVALS)))[:total_intervals]

def solve_mpc_horizon(ld_win, pv_win, start_soc, cap, b_lim, up_lim, lo_lim, solver, eta_win, wt):
    ld_win, pv_win, eta_win, len_h = np.asarray(ld_win).flatten(), np.asarray(pv_win).flatten(), np.asarray(eta_win).flatten(), len(ld_win)
    pb, pg = cp.Variable(len_h), cp.Variable(len_h)
    soc = start_soc - cp.cumsum(HR_STEP * pb)
    prob = cp.Problem(cp.Minimize(cp.sum(-HR_STEP * cp.multiply(eta_win, pb) + wt * cp.multiply(eta_win, cp.square(pg)))), 
                      [pg == ld_win - pv_win - pb, pb <= b_lim, pb >= -b_lim, soc >= 0.0, soc <= cap, soc[-1] == start_soc, pg <= up_lim, pg >= lo_lim])
    prob.solve(solver=getattr(cp, solver.upper()), verbose=False)
    return np.asarray(pb.value).flatten(), np.asarray(pg.value).flatten(), np.asarray(soc.value).flatten(), str(prob.status), float(prob.value)

def disaggregate_battery(agg_batt):
    return {n: float(agg_batt) * v["customers"] / MAX_CUSTOMERS for n, v in NODE_CONFIG.items()}

class ModbusConnection:
    def __init__(self, ip, port): self.ip, self.port, self.client = ip, port, None
    def connect(self):
        for _ in range(CONN_RETRIES):
            try:
                self.client = ModbusTcpClient(self.ip, port=self.port)
                if self.client.connect(): return
            except: pass
            time.sleep(CONN_DELAY_S)
        raise ConnectionError(f"Cannot connect to {self.ip}:{self.port}")
    def reconnect(self): self.close(); self.connect()
    def close(self):
        try: self.client.close() if self.client else None
        except: pass
        self.client = None

def modbus_write_node(conn, reg_start, p_load, q_ref, p_pv, p_batt, order, dry_run, verbose=False):
    vals = [p_load, q_ref, p_pv, p_batt] if order == "normal" else [p_batt, p_pv, q_ref, p_load]
    payload = [(int(round(float(v))) & 0xFFFF) if MIN_SIGNED_16BIT <= int(round(float(v))) <= MAX_SIGNED_16BIT else (max(MIN_SIGNED_16BIT, min(MAX_SIGNED_16BIT, int(round(float(v))))) & 0xFFFF) for v in vals]
    if dry_run: return print(f"    [DRY-RUN] holding {reg_start}-{reg_start+3}: {payload}") if verbose else None
    for attempt in range(2):
        try:
            if conn.client.write_registers(address=reg_start, values=payload).isError(): raise IOError
            return
        except: conn.reconnect() if attempt == 0 else print("ERROR: write failed after reconnect.")

def clear_registers(conn, dry_run):
    if dry_run: return print(f"[DRY-RUN] clear holding {REG_BEGIN}-{REG_BEGIN+REG_NUM-1}")
    if conn.client.write_registers(address=REG_BEGIN, values=[0]*REG_NUM).isError(): print("WARNING: could not clear holding registers.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default="central_agg_forecast_data_students.csv"); p.add_argument("--actual", default="agg_jan2013_students.csv")
    p.add_argument("--w", type=float, default=STD_W); p.add_argument("--horizon-steps", type=int, default=PREDICTION_STEPS)
    p.add_argument("--playback-steps", type=int, default=SIM_STEPS); p.add_argument("--step-seconds", type=float, default=SEC_PER_STEP)
    p.add_argument("--batt-power", type=float, default=TOT_POWER_KW); p.add_argument("--capacity", type=float, default=TOT_CAPACITY_KWH)
    p.add_argument("--initial_soc", type=float, default=START_SOC_KWH); p.add_argument("--p-upper", type=float, default=UPPER_P_KW)
    p.add_argument("--p-lower", type=float, default=LOWER_P_KW); p.add_argument("--solver", default="OSQP")
    p.add_argument("--ip", default=DEF_HIL_IP); p.add_argument("--port", type=int, default=DEF_HIL_PORT)
    for flag in ["--dry-run", "--no-wait", "--no-prompt", "--verbose", "--keep-final", "--no-soc"]: p.add_argument(flag, action="store_true")
    args = p.parse_args()

    f_ld, f_pv = load_forecast(Path(args.forecast))
    act_data, act_steps = load_actual_node_profiles(Path(args.actual))
    eta = generate_eta_profiles(args.playback_steps + args.horizon_steps - 1)
    
    mb = ModbusConnection(args.ip, args.port) if not args.dry_run else None
    if mb: mb.connect()
    
    if not args.no_prompt and not args.dry_run: input("\nPress Enter to start Experiment 2...")
    clear_registers(mb, args.dry_run)
    if not args.no_wait and not args.dry_run: time.sleep(12)

    rows, sim_soc, aborted = [], float(args.initial_soc), False
    try:
        for t in range(args.playback_steps):
            t0 = time.time()
            pb_seq, pg_seq, _, stat, obj = solve_mpc_horizon(f_ld[t:t+args.horizon_steps], f_pv[t:t+args.horizon_steps], sim_soc, args.capacity, args.batt_power, args.p_upper, args.p_lower, args.solver, eta[t:t+args.horizon_steps], args.w)
            agg_batt, fcst_grid, tot_ld, tot_pv, split_batt = float(pb_seq[0]), float(pg_seq[0]), 0.0, 0.0, disaggregate_battery(float(pb_seq[0]))
            
            for n in NODE_CONFIG:
                ld_val, pv_val = float(act_data[n]["load"][t]), float(act_data[n]["pv"][t])
                tot_ld += ld_val; tot_pv += pv_val
                modbus_write_node(mb, NODE_CONFIG[n]["register"], ld_val, NODE_CONFIG[n]["qref"], pv_val, split_batt[n], NODE_CONFIG[n]["order"], args.dry_run, args.verbose)
            
            soc_bef, sim_soc = sim_soc, max(0.0, min(args.capacity, sim_soc - HR_STEP * agg_batt))
            meas_soc = (mb.client.read_input_registers(address=SOC_INP_REG, count=1).registers[0] / 100.0) if mb and not args.no_soc else None
            
            r = {"step": t+1, "Pload_forecast_kw": f_ld[t], "PPV_forecast_kw": f_pv[t], "Pload_actual_feeder_kw": tot_ld, "PPV_actual_feeder_kw": tot_pv, "baseline_grid_kw": tot_ld-tot_pv, "Pbat_aggregate_kw": agg_batt, "X2_grid_actual_kw": tot_ld-tot_pv-agg_batt, "X2_grid_forecast_kw": fcst_grid, "soc_before_pct": 100.0*soc_bef/args.capacity, "soc_predicted_pct": 100.0*sim_soc/args.capacity, "soc_measured_pct": meas_soc or "", "battery_action": _action(agg_batt), "solver_status": stat, "objective_value": obj}
            r.update({f"{n}_Pbat_kw": round(split_batt[n], 6) for n in NODE_CONFIG})
            rows.append(r)
            
            if args.verbose or t == 0 or t == args.playback_steps - 1 or (t+1) % 10 == 0:
                print(f"Step {t+1:3d}/{args.playback_steps}: Load={tot_ld:8.1f} | PV={tot_pv:8.1f} | Batt={agg_batt:8.1f} | Grid={tot_ld-tot_pv-agg_batt:8.1f} | SoC={100*sim_soc/args.capacity:6.2f}%")
            if not args.no_wait and not args.dry_run: time.sleep(max(0.0, args.step_seconds - (time.time() - t0)))
    except KeyboardInterrupt: aborted = True
    finally:
        if args.keep_final: print("\nFinal HIL values kept.")
        else: clear_registers(mb, args.dry_run)
        if mb: mb.close()

    if rows: pd.DataFrame(rows).to_csv(f"experiment2_mpc_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print("\nExperiment 2 finished." + (" (aborted)" if aborted else ""))

if __name__ == "__main__": main()