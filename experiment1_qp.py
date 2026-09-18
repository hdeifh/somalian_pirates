#!/usr/bin/env python3
"""
ECE4191 Module 3 -- 16-node Toy QP over Modbus TCP
===================================================

A single-node (the node-phase groups) reduction of ``central_qp_modbus.py`` for
"triangle / square" unit test.

Input CSV formats
------------------
* ``--input`` (forecast, aggregate feeder): long-format CSV with columns
  ``Date, N_Customers, Profile, <48 half-hour time-of-day columns>``.
  Two rows per date -- one with ``Profile == "GC_Load_kW"`` and one with
  ``Profile == "PV_Generation_kW"`` -- giving the aggregate load/PV forecast
  for that day (e.g. ``central_agg_forecast_data_students.csv``).

* ``--actual`` (actual, per node-phase): long-format CSV with columns
  ``Date, Node, Phase, N_Customers, Profile, <48 half-hour time-of-day
  columns>``. Two rows per (Date, Node, Phase) -- ``GC_Load_kW`` and
  ``PV_Generation_kW`` -- giving the actual load/PV for that node-phase group
  on that day (e.g. ``agg_jan2013_students.csv``). ``Phase`` may be given
  either as ``Ph1``/``Ph2``/``Ph3`` or directly as ``A``/``B``/``C``; both are
  mapped onto the ``A``/``B``/``C`` suffixes used in the ``NODES`` dict below
  (``Ph1`` -> ``A``, ``Ph2`` -> ``B``, ``Ph3`` -> ``C``).

* the node-phase groups: 102 customers -> 1020 kWh battery capacity.
  Initial SoC = 50% = 510 kWh. Battery power limit default 510 kW
  (5 kW/customer; override with --batt-power).

QP (per day, replayed over 48 steps)
------------------------------------
    variables : batt (48), grid (48)
    minimize    sum_squares(grid)                 # "minimize x^2 only"
    subject to
        grid == P_load - P_PV - batt              # A2 x = b2  (equality)
        sum(batt) == 0                            # A2 x = b2  (daily net-zero)
        -Pb <= batt <= Pb                         # A1 x <= b1 (inequality)
        0   <= soc0 - cumsum(batt*dt) <= C        # A1 x <= b1 (SoC bounds)
        [ p_lower <= grid <= p_upper ]            # A1 x <= b1 (optional feeder)

Forecast vs actual (this variant)
---------------------------------
* The QP is solved on the FORECAST CSV (``--input``): it minimises
  sum(x2^2) with x2 = P_load_hat - P_PV_hat - batt, producing the battery
  command X1 = batt. The forecast grid X2 from the QP is discarded.
* The ACTUAL CSV (``--actual``) supplies the load/PV that are written to
  the node-phase groups, and the optimised grid power is recomputed as
      X2 = P_load - P_PV - batt      (actual load/PV, forecast batt command).

Battery sign convention (unchanged from the Modbus reference):
    batt > 0  = discharge  ->  grid = load - pv - batt.
    Registers hold direct kW/kVAr (do NOT multiply by 1000). No sign flip.

AI Declaration - AI was used to write the code, with human-inputted promopts (AI-written parts are commented)
"""
from __future__ import annotations
import argparse, sys, time, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

try: import cvxpy as cp
except ImportError: sys.exit("ERROR: cvxpy not found. Install it with:\n  pip install cvxpy --break-system-packages")
try: from pymodbus.client import ModbusTcpClient
except ImportError: sys.exit("ERROR: pymodbus not found. Install it with:\n  pip install pymodbus --break-system-packages")

INTV_STEPS, HR_DELTA, SEC_STEP, OBJ_WT = 48, 0.5, 2.0, 1.0
ND_CFG = {
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
TOT_CUST = sum(n["customers"] for n in ND_CFG.values())
TOT_CAP_KWH, TOT_PWR_KW, START_SOC_KWH = 10.0 * TOT_CUST, 5.0 * TOT_CUST, 0.5 * 10.0 * TOT_CUST
for k, v in ND_CFG.items(): v.update({"capacity_kwh": 10.0 * v["customers"], "power_kw": 5.0 * v["customers"], "initial_soc_kwh": 5.0 * v["customers"]})
UP_LIM_KW, LO_LIM_KW = 3000, -1500
DEF_IP, DEF_PORT, REG_START, REG_CNT, SOC_REG = "192.168.1.210", 502, 2000, 64, 3000
MIN_S16, MAX_S16, RETRIES, DELAY_S = -32768, 32767, 5, 2.0
PH_MAP = {"Ph1": "A", "Ph2": "B", "Ph3": "C", "A": "A", "B": "B", "C": "C"}

'''
AI Declaration: _get_t_cols, load_fcst_data, load_act_data, run_qp_day, write_mb_node, wipe_regs were done with AI assistance
ModBusClient and SocReader classes were written by ECE4191 teaching staff (re-used from other toy_qp and toy_mpc files)
'''

def _get_t_cols(cols):
    """Return the ordered list of half-hour time-of-day columns in a CSV,
    excluding the metadata columns and any stray 'Unnamed: N' columns left
    behind by blank header cells."""
    return [c for c in cols if c not in {"Date", "Node", "Phase", "N_Customers", "Profile"} and not str(c).startswith("Unnamed")]

def load_fcst_data(fpath: Path):
    """Read the long-format aggregate forecast CSV and return per-day PV /
    load arrays for the whole feeder."""
    df = pd.read_csv(fpath); df.columns = [str(c).strip() for c in df.columns]
    tc = _get_t_cols(df.columns); df["_dt"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    dts = sorted(df["_dt"].dropna().unique())
    pv_ls = [pd.to_numeric(df[(df["_dt"] == d) & df["Profile"].str.contains("PV", case=False)].iloc[0][tc], errors="coerce").to_numpy(float) for d in dts]
    ld_ls = [pd.to_numeric(df[(df["_dt"] == d) & df["Profile"].str.contains("Load", case=False)].iloc[0][tc], errors="coerce").to_numpy(float) for d in dts]
    return np.vstack(pv_ls), np.vstack(ld_ls), len(dts)

def load_act_data(fpath: Path):
    """Read the long-format per-node-phase actual CSV and return per-day PV /
    load arrays for every node-phase group in NODES."""
    df = pd.read_csv(fpath); df.columns = [str(c).strip() for c in df.columns]
    tc, df["_dt"] = _get_t_cols(df.columns), pd.to_datetime(df["Date"], format="%d-%b-%y")
    df["_nk"] = df["Node"].astype(str).str.strip() + "_" + df["Phase"].astype(str).str.strip().map(PH_MAP)
    dts, act_pv, act_ld = sorted(df["_dt"].dropna().unique()), {}, {}
    for nk in ND_CFG:
        nr = df[df["_nk"] == nk]
        act_ld[nk] = np.vstack([pd.to_numeric(nr[(nr["_dt"] == d) & nr["Profile"].str.contains("Load", case=False)].iloc[0][tc], errors="coerce").to_numpy(float) for d in dts])
        act_pv[nk] = np.vstack([pd.to_numeric(nr[(nr["_dt"] == d) & nr["Profile"].str.contains("PV", case=False)].iloc[0][tc], errors="coerce").to_numpy(float) for d in dts])
    return act_pv, act_ld, len(dts)

def run_qp_day(ld_kw, pv_kw, b_lim, cap, start_soc, up_lim, lo_lim, solver_nm):
    """Single-node load-levelling QP for one day.
    Objective : minimize ||grid||^2      (pure quadratic, "x^2 only")
    Equality  : grid = load - pv - batt ,  sum(batt) = 0        (A2 x = b2)
    Inequality: battery power, SoC bounds, optional feeder bounds (A1 x <= b1)
    """
    pb, pg, eta = cp.Variable(len(ld_kw)), cp.Variable(len(ld_kw)), np.zeros(INTV_STEPS)
    eta[np.r_[0:14, 44:48]], eta[np.r_[14:28, 40:44]], eta[28:40] = 0.03, 0.06, 0.30
    soc = start_soc - cp.cumsum(pb * HR_DELTA)
    prob = cp.Problem(cp.Minimize(cp.sum(-HR_DELTA * cp.multiply(eta, pb) + OBJ_WT * cp.multiply(eta, cp.square(pg)))), 
                      [pg == ld_kw - pv_kw - pb, cp.sum(pb) == 0, pb <= b_lim, pb >= -b_lim, soc >= 0.0, soc <= cap, pg <= up_lim, pg >= lo_lim])
    prob.solve(solver=getattr(cp, solver_nm), verbose=False)
    return np.asarray(pb.value, float).flatten(), np.asarray(pg.value, float).flatten(), str(prob.status), float(prob.value)

class ModbusClient:
    """Live ModbusTcpClient with reconnect-on-failure (from the v7/v8 logic)."""
    def __init__(self, ip, port): self.ip, self.port, self.conn = ip, port, None
    def connect(self):
        for _ in range(RETRIES):
            try:
                self.conn = ModbusTcpClient(self.ip, port=self.port)
                if self.conn.connect(): return
            except: pass
            time.sleep(DELAY_S)
        raise ConnectionError(f"Cannot connect to {self.ip}:{self.port}")
    def reconnect(self): self.close(); self.connect()
    def close(self):
        try: self.conn.close() if self.conn else None
        except: pass
        self.conn = None

def write_mb_node(mb_cli, reg_start, p_ld, p_q, p_pv, p_bt, order, is_dry, verbose=False):
    """Write the 4 registers for one node-phase."""
    vals = [p_ld, p_q, p_pv, p_bt] if order == "normal" else [p_bt, p_pv, p_q, p_ld]
    payload = [(int(round(v)) & 0xFFFF) if MIN_S16 <= int(round(v)) <= MAX_S16 else (max(MIN_S16, min(MAX_S16, int(round(v)))) & 0xFFFF) for v in vals]
    if is_dry: return print(f"    [DRY-RUN] holding[{reg_start}:{reg_start + 3}] = {payload}") if verbose else None
    for attempt in range(2):
        try:
            if mb_cli.conn.write_registers(address=reg_start, values=payload).isError(): raise IOError
            return
        except: mb_cli.reconnect() if attempt == 0 else print("  ERROR: Modbus write failed again after reconnection.")

def wipe_regs(mb_cli, is_dry):
    if is_dry: return print(f"    [DRY-RUN] clear holding[{REG_START}:{REG_START+REG_CNT}]")
    if mb_cli.conn.write_registers(address=REG_START, values=[0]*REG_CNT).isError(): print(f"  WARNING: failed to clear registers {REG_START}-{REG_START + REG_CNT - 1}.")

class SocReader:
    """Read 16-node SoC (input register 3000, value/100) each step and return
    it so callers can fold it into the schedule. Does NOT write a standalone
    SoC CSV -- the measured SoC lives only in toy_qp_schedule_<ts>.csv."""
    def __init__(self, mb_cli, en): self.mb_cli, self.en = mb_cli, en
    def start(self):
        if not self.en or not self.mb_cli or not self.mb_cli.conn: return setattr(self, 'en', False)
        try: self.mb_cli.conn.read_input_registers(address=SOC_REG, count=1).registers[0] / 100.0
        except: print("  WARNING: SoC register did not read back; soc_measured_pct will be blank.")
    def read_step(self):
        if not self.en: return None
        try: return self.mb_cli.conn.read_input_registers(address=SOC_REG, count=1).registers[0] / 100.0
        except: return None

def main():
    p = argparse.ArgumentParser(description="ECE4191 Module 3 -- Aggregated QP + 16-node Modbus playback")
    for f, d in [("--input", "central_agg_forecast_data_students.csv"), ("--actual", "agg_jan2013_students.csv"), ("--solver", "OSQP"), ("--ip", DEF_IP)]: p.add_argument(f, default=d)
    for f, d in [("--step-seconds", SEC_STEP), ("--batt-power", TOT_PWR_KW), ("--capacity", TOT_CAP_KWH), ("--initial_soc", START_SOC_KWH), ("--p-upper", UP_LIM_KW), ("--p-lower", LO_LIM_KW), ("--port", DEF_PORT), ("--measurement-delay", 0.2)]: p.add_argument(f, type=float if isinstance(d, float) else int, default=d)
    p.add_argument("--progress-every", type=int, default=1)
    for flag in ["--no-wait", "--no-prompt", "--dry-run", "--keep-final", "--no-measurements", "--verbose"]: p.add_argument(flag, action="store_true")
    args = p.parse_args()

    mb_cli = ModbusClient(args.ip, args.port) if not args.dry_run else None
    if mb_cli: mb_cli.connect()
    
    fcst_pv, fcst_ld, fcst_days = load_fcst_data(Path(args.input))
    act_pv, act_ld, act_days = load_act_data(Path(args.actual))
    
    if not args.no_prompt and not args.dry_run: input("\nPress Enter to start playback ...\n")
    wipe_regs(mb_cli, args.dry_run)
    if not args.no_wait and not args.dry_run: time.sleep(12)

    soc_rdr, out_rs, abort = SocReader(mb_cli, not args.no_measurements and not args.dry_run), [], False
    soc_rdr.start()

    try:
        for d in range(fcst_days):
            pb_kw, pg_kw, stat, obj = run_qp_day(fcst_ld[d], fcst_pv[d], args.batt_power, args.capacity, args.initial_soc, args.p_upper, args.p_lower, args.solver)
            act_ld_agg, act_pv_agg = sum(act_ld[n][d] for n in ND_CFG), sum(act_pv[n][d] for n in ND_CFG)
            act_pg_agg, soc_traj = act_ld_agg - act_pv_agg - pb_kw, args.initial_soc - np.cumsum(pb_kw * HR_DELTA)
            bt_nds = {n: pb_kw * (ND_CFG[n]["customers"] / TOT_CUST) for n in ND_CFG}

            for k in range(INTV_STEPS):
                t0, stp = time.time(), d * INTV_STEPS + k + 1
                for n, inf in ND_CFG.items():
                    write_mb_node(mb_cli, inf["register"], act_ld[n][d][k], inf["qref"], act_pv[n][d][k], bt_nds[n][k], inf["order"], args.dry_run, args.verbose)
                if not args.no_wait and args.measurement_delay > 0 and not args.dry_run: time.sleep(args.measurement_delay)
                m_soc, p_soc = soc_rdr.read_step(), 100.0 * float(soc_traj[k]) / args.capacity
                
                r = {"step": stp, "p_load_fc_kw": fcst_ld[d][k], "p_pv_fc_kw": fcst_pv[d][k], "p_load_kw": act_ld_agg[k], "p_pv_kw": act_pv_agg[k], "baseline_grid_kw": act_ld_agg[k]-act_pv_agg[k], "X1_battery_kw": pb_kw[k], "battery_action": "Discharge" if pb_kw[k]>0.5 else ("Charge" if pb_kw[k]<-0.5 else "Idle"), "X2_grid_kw": act_pg_agg[k], "X2_grid_forecast_kw": pg_kw[k], "soc_predicted_pct": p_soc, "soc_measured_pct": m_soc or ""}
                r.update({f"{n}_batt_kw": bt_nds[n][k] for n in ND_CFG}); out_rs.append(r)
                
                if not args.no_wait and not args.dry_run: time.sleep(max(0.0, args.step_seconds - (time.time() - t0)))
    except KeyboardInterrupt: abort = True
    finally:
        if args.keep_final: print("\nFinal HIL state kept (--keep-final).")
        else: wipe_regs(mb_cli, args.dry_run)
        if mb_cli: mb_cli.close()

    if out_rs: pd.DataFrame(out_rs).to_csv(f"central_qp_16node_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print("\nPlayback complete." + ("  (aborted)" if abort else ""))

if __name__ == "__main__": main()