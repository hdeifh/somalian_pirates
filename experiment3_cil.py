#!/usr/bin/env python3
"""
ECE4191 Module 3 -- Experiment 3: MPC-CIL at Node 646 over Modbus TCP
=======================================================================

Controller-in-the-Loop (CIL) MPC for the battery at Node 646 (Phase B).

Key differences from Experiment 2 (feeder-wide MPC):
  - Only Node 646's battery is controlled; ALL other node battery commands = 0.
  - Node 646 battery parameters are per the toy script: C=1020 kWh, Pb=510 kW.
  - Load/PV for ALL 16 nodes is still written to the HIL each step (full feeder
    remains in operation; only the battery register differs).
  - Both Node 646 MPC forecast and all 16 nodes' actual load/PV come from 
    agg_jan2013_students.csv.
  - The measured SoC from register 3000 feeds directly into the next horizon
    solve (true CIL closed-loop feedback).
  - 4-day playback (192 steps), same as Experiment 2.

Node 646 battery parameters (Module 3, p.29):
  Capacity  C   = 1020 kWh   (102 customers x 10 kWh)
  Power     Pb  = 510 kW     (102 customers x 5 kW)
  Init SoC       = 510 kWh   (50%)
  Qref           = 132 kVAr  (fixed)

MPC Objective (Module 2 formulation, as required by Exp 3):
  minimize  sum( -delta * eta(k) * x1(k)  +  w * eta(k) * x2(k)^2 )
  where eta(k) = time-of-use tariff, w = 1, delta = 0.5

Forecasts:
  Node 646 forecast uses agg_jan2013_students.csv (240 steps).
    - Horizon window [t : t+48] is sliced from the 240-step array.
    - Playback = 240 - 48 = 192 steps (4 days).

Register map (Table 4, Module 3 p.18):
  Normal  nodes: [Pload, Qload, PPV, Pbat]
  Reversed nodes: [Pbat, PPV, Qload, Pload]
  Pbat for all nodes except 646_B is written as 0.
"""
from __future__ import annotations
import argparse, sys, time, re, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

try: import cvxpy as cp
except ImportError: sys.exit("ERROR: cvxpy not found. Install it with:\n  pip install cvxpy --break-system-packages")
try: from pymodbus.client import ModbusTcpClient
except ImportError: sys.exit("ERROR: pymodbus not found. Install it with:\n  pip install pymodbus --break-system-packages")

N_INTV, HR_DELTA, SEC_STEP, HRZ_DEF = 48, 0.5, 2.0, 48
CAP_646, PWR_646, SOC0_646, QRF_646, OBJ_WT = 10.0 * 102, 5.0 * 102, 0.5 * 10.0 * 102, 132.0, 1.0
UP_KW, LO_KW = 3000, -1500
ND_CUST = {"646_B": 102, "645_B": 63, "611_C": 68, "652_A": 46, "671_A": 159, "671_B": 155, "671_C": 159, "692_C": 66, "692_B": 0, "692_A": 0, "675_C": 119, "675_B": 36, "675_A": 191, "634_C": 52, "634_B": 45, "634_A": 69}
REV_NDS, ND_KYS = {"692_C", "692_B", "692_A", "675_C", "675_B", "675_A", "634_C", "634_B", "634_A"}, list(ND_CUST.keys())
QRF_DICT = {k: QRF_646 if k == "646_B" else 0.0 for k in ND_KYS}
DEF_IP, DEF_PORT, REG_STRT, REG_CNT, REGS_PN = "192.168.1.210", 502, 2000, 64, 4
MIN_S16, MAX_S16, SOC_REG, RETRIES, DELAY_S = -32768, 32767, 3000, 5, 2.0
FCST_CSV_DEF, ACT_CSV_DEF = "agg_jan2013_students.csv", "agg_jan2013_students.csv"

'''
AI Declaration: load_data, solve_cil_horizon were written with AI assistance.
ModBusClient and SocReader classes were written by ECE4191 teaching staff (re-used from other toy_qp and toy_mpc files)
'''

def gen_eta():
    """Return 48-step daily time-of-use tariff profile ($/kWh).
    Matches Module 2 generate_eta_profiles():
      Off-peak  0.03  steps  0-13  (midnight-7am)  and 44-47 (10pm-midnight)
      Shoulder  0.06  steps 14-27  (7am-2pm)       and 40-43 (8pm-10pm)
      Peak      0.30  steps 28-39  (2pm-8pm)
    """
    eta = np.zeros(N_INTV)
    eta[np.r_[0:14, 44:48]], eta[np.r_[14:28, 40:44]], eta[28:40] = 0.03, 0.06, 0.30
    return eta

def load_data(fpath: Path):
    """Load agg_jan2013_students.csv per-node-phase actual load/PV."""
    df = pd.read_csv(fpath); tc = [c for c in df.columns if re.match(r'^\d+:\d+$', str(c).strip())]
    df["_nk"] = df["Node"].astype(str).str.strip() + "_" + df["Phase"].apply(lambda p: {"Ph1": "A", "Ph2": "B", "Ph3": "C"}.get(str(p).strip(), str(p).strip().upper()))
    dts = sorted(df["Date"].unique(), key=lambda x: pd.to_datetime(x, format="%d-%b-%y"))
    act_ld, act_pv = {k: np.zeros((len(dts), N_INTV)) for k in ND_KYS}, {k: np.zeros((len(dts), N_INTV)) for k in ND_KYS}
    for k in ND_KYS:
        nr = df[df["_nk"] == k]
        for i, d in enumerate(dts):
            dr = nr[nr["Date"] == d]
            lr, pr = dr[dr["Profile"].str.contains("Load", case=False, na=False)], dr[dr["Profile"].str.contains("PV|Generation", case=False, na=False)]
            if not lr.empty: act_ld[k][i] = pd.to_numeric(lr[tc].iloc[0], errors="coerce").to_numpy(float)
            if not pr.empty: act_pv[k][i] = pd.to_numeric(pr[tc].iloc[0], errors="coerce").to_numpy(float)
    return act_ld, act_pv, len(dts)

def solve_cil_horizon(ld_win, pv_win, eta_win, s0, cap, b_pwr, up_pwr, lo_pwr, w, slvr):
    """Solve the Module 2 tariff-weighted QP over one look-ahead window.
    Objective (Module 2, Task 1):
        minimize  sum( -delta * eta(k) * x1(k)  +  w * eta(k) * x2(k)^2 )
    NOTE: No terminal SoC constraint (D-RHO online procedure).
    """
    pb, pg = cp.Variable(len(ld_win)), cp.Variable(len(ld_win))
    soc = s0 - cp.cumsum(pb * HR_DELTA)
    prob = cp.Problem(cp.Minimize(cp.sum(-HR_DELTA * (eta_win @ pb) + w * (eta_win @ cp.square(pg)))),
                      [pg == ld_win - pv_win - pb, pb <= b_pwr, pb >= -b_pwr, soc >= 0.0, soc <= cap, pg <= up_pwr, pg >= lo_pwr])
    prob.solve(solver=getattr(cp, slvr), verbose=False)
    return (np.asarray(pb.value, float).flatten(), np.asarray(pg.value, float).flatten(), str(prob.status)) if pb.value is not None else (np.zeros(len(ld_win)), ld_win - pv_win, "solver_failed")

class ModbusClient:
    """Live ModbusTcpClient with reconnect-on-failure."""
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

def write_all_nds(mb_cli, plds, is_dry, verbose=False):
    """Write all 16 nodes as a single 64-register burst (2000-2063)."""
    flat = [v for k in ND_KYS for v in plds[k]]
    if is_dry: return print(f"    [DRY-RUN] regs {REG_STRT}-{REG_STRT+REG_CNT-1} = {flat[:10]}...") if verbose else None
    for attempt in range(2):
        try:
            if mb_cli.conn.write_registers(address=REG_STRT, values=flat).isError(): raise IOError
            return
        except: mb_cli.reconnect() if attempt == 0 else print("  ERROR: Modbus write still failed after reconnect. Step dropped.")

def wipe_regs(mb_cli, is_dry):
    if is_dry: return print(f"    [DRY-RUN] clear holding[{REG_STRT}:{REG_STRT+REG_CNT}]")
    if mb_cli.conn.write_registers(address=REG_STRT, values=[0]*REG_CNT).isError(): print(f"  WARNING: failed to clear registers {REG_STRT}-{REG_STRT + REG_CNT - 1}.")

class SocReader:
    """Read Node-646 SoC (input register 3000) as CIL closed-loop feedback."""
    def __init__(self, mb_cli, en): self.mb_cli, self.en = mb_cli, en
    def start(self):
        if not self.en or not self.mb_cli or not self.mb_cli.conn: return setattr(self, 'en', False)
        try: self.mb_cli.conn.read_input_registers(address=SOC_REG, count=1).registers[0] / 100.0
        except: print("  WARNING: SoC register did not read back; model SoC will be used as fallback.")
    def read_pct(self, is_dry, sim_pct):
        if is_dry: return sim_pct
        if not self.en or not self.mb_cli or not self.mb_cli.conn: return None
        for attempt in range(2):
            try: return self.mb_cli.conn.read_input_registers(address=SOC_REG, count=1).registers[0] / 100.0
            except: self.mb_cli.reconnect() if attempt == 0 else print("  ERROR: SoC re-read failed.")
        return None

def main():
    p = argparse.ArgumentParser(description="ECE4191 Module 3 Exp 3 -- MPC-CIL at Node 646 over Modbus TCP")
    for f, d in [("--n646-forecast", FCST_CSV_DEF), ("--actual", ACT_CSV_DEF), ("--solver", "OSQP"), ("--ip", DEF_IP)]: p.add_argument(f, default=d)
    for f, d in [("--horizon-steps", HRZ_DEF), ("--batt-power", PWR_646), ("--capacity", CAP_646), ("--initial-soc", SOC0_646), ("--weight", OBJ_WT), ("--p-upper", UP_KW), ("--p-lower", LO_KW), ("--step-seconds", SEC_STEP), ("--port", DEF_PORT), ("--measurement-delay", 0.2)]: p.add_argument(f, type=float if isinstance(d, float) else int, default=d)
    p.add_argument("--progress-every", type=int, default=1)
    for flag in ["--dry-run", "--no-wait", "--no-prompt", "--no-measurements", "--verbose", "--keep-final"]: p.add_argument(flag, action="store_true")
    args = p.parse_args()

    mb_cli = ModbusClient(args.ip, args.port) if not args.dry_run else None
    if mb_cli: mb_cli.connect()

    a_ld, a_pv, _ = load_data(Path(args.n646_forecast)); fc_ld, fc_pv = a_ld["646_B"].flatten(), a_pv["646_B"].flatten()
    tot_fc = 5 * N_INTV; reps = int(np.ceil(tot_fc / len(fc_ld)))
    fc_ld, fc_pv, pb_stps = np.tile(fc_ld, reps)[:tot_fc], np.tile(fc_pv, reps)[:tot_fc], tot_fc - args.horizon_steps
    eta_flat = np.tile(gen_eta(), int(np.ceil(tot_fc / N_INTV)))[:tot_fc]

    act_ld, act_pv, _ = load_data(Path(args.actual))
    flat_ld, flat_pv = {k: act_ld[k].flatten() for k in ND_KYS}, {k: act_pv[k].flatten() for k in ND_KYS}

    if not args.no_prompt and not args.dry_run: input("\nPress Enter to start playback ...\n")
    wipe_regs(mb_cli, args.dry_run)
    if not args.no_wait and not args.dry_run: time.sleep(12)

    soc_rdr, out_rs, abort, mod_soc = SocReader(mb_cli, not args.no_measurements and not args.dry_run), [], False, None
    soc_rdr.start()

    try:
        for t in range(pb_stps):
            t0, stp = time.time(), t + 1
            m_pct = soc_rdr.read_pct(args.dry_run, 100.0 * (args.initial_soc if mod_soc is None else mod_soc) / args.capacity)
            s0 = max(0.0, min(args.capacity, m_pct / 100.0 * args.capacity)) if m_pct is not None else (args.initial_soc if mod_soc is None else mod_soc)
            if mod_soc is None: mod_soc = s0
            
            pb_seq, _, stat = solve_cil_horizon(fc_ld[t:min(t+args.horizon_steps, tot_fc)], fc_pv[t:min(t+args.horizon_steps, tot_fc)], eta_flat[t:min(t+args.horizon_steps, tot_fc)], s0, args.capacity, args.batt_power, args.p_upper, args.p_lower, args.weight, args.solver)
            b0 = float(pb_seq[0])

            l_act, p_act = float(flat_ld["646_B"][t]), float(flat_pv["646_B"][t])
            plds = {k: [(int(round(v)) & 0xFFFF) if MIN_S16 <= int(round(v)) <= MAX_S16 else (max(MIN_S16, min(MAX_S16, int(round(v)))) & 0xFFFF) for v in ([float(flat_ld[k][t]), QRF_DICT[k], float(flat_pv[k][t]), b0 if k == "646_B" else 0.0] if k not in REV_NDS else [b0 if k == "646_B" else 0.0, float(flat_pv[k][t]), QRF_DICT[k], float(flat_ld[k][t])])] for k in ND_KYS}
            
            write_all_nds(mb_cli, plds, args.dry_run, args.verbose)
            if not args.no_wait and args.measurement_delay > 0 and not args.dry_run: time.sleep(args.measurement_delay)

            out_rs.append({"step": stp, "day": (t // N_INTV) + 1, "step_of_day": (t % N_INTV) + 1, "eta_kw_per_kwh": eta_flat[t], "p_load_fc_n646_kw": fc_ld[t], "p_pv_fc_n646_kw": fc_pv[t], "p_load_act_n646_kw": l_act, "p_pv_act_n646_kw": p_act, "baseline_grid_n646_kw": l_act-p_act, "X1_battery_n646_kw": b0, "battery_action": "Discharge" if b0>0.5 else ("Charge" if b0<-0.5 else "Idle"), "X2_grid_n646_actual_kw": l_act-p_act-b0, "X2_grid_n646_fc_kw": fc_ld[t]-fc_pv[t]-b0, "soc_predicted_pct": 100.0*mod_soc/args.capacity, "soc_measured_pct": m_pct or ""})
            mod_soc = max(0.0, min(args.capacity, mod_soc - b0 * HR_DELTA))
            if not args.no_wait and not args.dry_run: time.sleep(max(0.0, args.step_seconds - (time.time() - t0)))
    except KeyboardInterrupt: abort = True
    finally:
        if args.keep_final: print("\nFinal HIL state kept (--keep-final).")
        else: wipe_regs(mb_cli, args.dry_run)
        if mb_cli: mb_cli.close()

    if out_rs: pd.DataFrame(out_rs).to_csv(f"central_mpc_cil_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print("\nPlayback complete." + ("  (aborted)" if abort else ""))

if __name__ == "__main__": main()