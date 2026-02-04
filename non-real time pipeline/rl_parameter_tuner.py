#!/usr/bin/env python3
"""
rl_parameter_tuner.py

Cross-Entropy Method (CEM) tuner for MPC meta-parameters (simulation-only).

Purpose:
- Tune MPC hyperparameters (control-to-phase gain B, control penalty lambda_u, control-rate penalty lambda_du)
  to minimize simulated tremor amplitude while keeping control energy low.
- Intended to tune the MPC you built in mpc_controller.py without touching the low-level control loop.

Usage:
    python rl_parameter_tuner.py --predictions lstm_state_estimator_fixed_predictions.csv --iters 50

Dependencies:
    numpy, pandas, matplotlib (optional). mpc_controller.py in same folder is preferred.
    If mpc_controller is not available, a conservative phase-locked fallback evaluator is used.

Author: Generated for ISEF workflow
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import math
import sys
import json
from typing import Tuple, Dict

# Try to import the user's MPC runner
try:
    from mpc_controller import run_receding_mpc
    HAS_MPC_MODULE = True
except Exception:
    HAS_MPC_MODULE = False

# -------------------------
# Helper: fallback evaluator (phase-locked stim simulation)
# -------------------------
def fallback_phase_locked_eval(pred_csv: str,
                               B: float,
                               lambda_u: float,
                               lambda_du: float,
                               desired_offset: float = math.pi,
                               amp_effect_gain: float = 0.5,
                               horizon_seconds: float = 0.8,
                               dt_override: float | None = None) -> Dict:
    """
    Simple simulation that:
      - Loads predictions CSV (time_s, amp_pred, phase_pred)
      - Schedules stimulation pulses at phase crossings offset by desired_offset
      - Stim amplitude is proportional to B (tunable)
      - Simulates amplitude suppression as multiplicative factor per pulse
    Returns a dict with aggregated metrics: mean_amp, total_control_energy, reward
    """
    df = pd.read_csv(pred_csv)
    t = df["time_s"].values.astype(float)
    amp = df["amp_pred"].values.astype(float)
    phase = df["phase_pred"].values.astype(float)

    # unwrap phase for consistency
    phi = np.unwrap(phase)
    if dt_override is None:
        dt = np.median(np.diff(t))
    else:
        dt = float(dt_override)

    # target angle
    target_angle = (desired_offset) % (2 * math.pi)
    inst_phase = np.mod(phi, 2 * math.pi)
    crossings = []
    for i in range(1, len(inst_phase)):
        if inst_phase[i-1] < target_angle <= inst_phase[i]:
            crossings.append(i)
    # limit worst-case
    if len(crossings) == 0:
        # no stimulation events; return baseline metrics
        mean_amp = float(np.mean(amp))
        return {"mean_amp": mean_amp, "total_energy": 0.0, "reward": -mean_amp}

    sim_amp = amp.copy()
    control_energy = 0.0
    for idx in crossings:
        # control magnitude scaled by B, clipped
        u = float(np.tanh(B))  # map B to bounded control value in (-1,1)
        control_energy += u*u
        supp = max(0.0, 1.0 - amp_effect_gain * abs(u))
        # apply immediate multiplicative attenuation onwards
        sim_amp[idx:] = sim_amp[idx:] * supp

    mean_amp = float(np.mean(sim_amp))
    # reward: we want to minimize amplitude and control energy
    reward = -mean_amp - 0.01 * control_energy
    return {"mean_amp": mean_amp, "total_energy": float(control_energy), "reward": float(reward)}


# -------------------------
# CEM optimizer
# -------------------------
def cem_optimize(evaluate_fn,
                 pred_csv: str,
                 param_names: list,
                 init_mean: np.ndarray,
                 init_std: np.ndarray,
                 n_iters: int = 40,
                 batch_size: int = 24,
                 elite_frac: float = 0.2,
                 save_prefix: str = "cem_tune"):
    """
    Generic CEM optimizer:
      - evaluate_fn(candidate_vector, pred_csv) -> metric (higher is better)
    Returns best_params, history
    """
    dim = len(init_mean)
    mean = init_mean.copy()
    std = init_std.copy()
    n_elite = max(1, int(np.round(batch_size * elite_frac)))

    history = []
    best = {"score": -np.inf, "params": None}

    for it in range(n_iters):
        samples = np.random.randn(batch_size, dim) * std[None, :] + mean[None, :]
        # enforce reasonable bounds (e.g., for lambda >= 0)
        # We'll clip in evaluate_fn if needed
        scores = np.zeros(batch_size)
        details = [None] * batch_size
        for i in range(batch_size):
            params = samples[i]
            try:
                res = evaluate_fn(pred_csv, *params) if callable(evaluate_fn) else evaluate_fn
                # if evaluate_fn expects vector, change accordingly
            except TypeError:
                # try passing vector as single arg then unpack
                res = evaluate_fn(pred_csv, *params)
            # res expected to be dict with 'reward' key
            if isinstance(res, dict) and "reward" in res:
                score = res["reward"]
            else:
                score = float(res)
            scores[i] = score
            details[i] = res

        # select elites
        idx_sorted = np.argsort(scores)[::-1]
        elites = samples[idx_sorted[:n_elite], :]
        elite_scores = scores[idx_sorted[:n_elite]]
        # update mean/std
        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0) + 1e-6
        # logging
        iter_best_idx = idx_sorted[0]
        iter_best_score = scores[iter_best_idx]
        iter_best_params = samples[iter_best_idx]
        print(f"[CEM] Iter {it+1}/{n_iters} | best_score {iter_best_score:.6f} | mean_reward {np.mean(scores):.6f}")
        history.append({
            "iter": it+1,
            "best_score": float(iter_best_score),
            "best_params": iter_best_params.tolist(),
            "mean_score": float(np.mean(scores))
        })
        if iter_best_score > best["score"]:
            best["score"] = float(iter_best_score)
            best["params"] = iter_best_params.copy()
            best["detail"] = details[iter_best_idx]
        # save intermediate results
        with open(f"{save_prefix}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        np.save(f"{save_prefix}_mean.npy", mean)
        np.save(f"{save_prefix}_std.npy", std)

    return best, history


# -------------------------
# Wrappers to evaluate parameter vectors (B, lambda_u, lambda_du)
# -------------------------
def evaluate_using_mpc(pred_csv: str,
                       B: float,
                       lambda_u: float,
                       lambda_du: float,
                       desired_offset: float = math.pi,
                       horizon_seconds: float = 0.8,
                       dt_override: float | None = None,
                       mpc_kwargs: dict = None) -> Dict:
    """
    Calls run_receding_mpc with provided meta-parameters and returns a dict with 'reward', 'mean_amp', 'total_energy'.
    Requires mpc_controller.run_receding_mpc to exist in import path.
    """
    if not HAS_MPC_MODULE:
        raise RuntimeError("mpc_controller not available for evaluate_using_mpc.")
    # delegate to run_receding_mpc; it expects B, lambda_u, lambda_du, desired offset etc.
    if mpc_kwargs is None:
        mpc_kwargs = {}
    res = run_receding_mpc(pred_csv,
                           out_csv="/tmp/rl_mpc_tmp_results.csv",
                           horizon_seconds=horizon_seconds,
                           dt_override=dt_override,
                           B=B,
                           u_min=-1.0,
                           u_max=1.0,
                           lambda_u=lambda_u,
                           lambda_du=lambda_du,
                           desired_phase_offset=desired_offset,
                           simulate_amp_effect=True,
                           amp_effect_gain=mpc_kwargs.get("amp_effect_gain", 0.5))
    # res should include amp_sim and u_applied arrays
    amp_sim = np.array(res["amp_sim"])
    u_applied = np.array(res["u_applied"])
    mean_amp = float(np.mean(amp_sim))
    total_energy = float(np.sum(u_applied**2))
    reward = -mean_amp - 0.01 * total_energy
    return {"mean_amp": mean_amp, "total_energy": total_energy, "reward": reward, "raw": res}


def evaluate_wrapper(pred_csv: str, B: float, lambda_u: float, lambda_du: float) -> Dict:
    """
    Wrapper that tries mpc first, falls back to phase-locked heuristic if unavailable.
    """
    try:
        if HAS_MPC_MODULE:
            return evaluate_using_mpc(pred_csv, B, lambda_u, lambda_du)
        else:
            return fallback_phase_locked_eval(pred_csv, B=B, lambda_u=lambda_u, lambda_du=lambda_du)
    except Exception as e:
        print(f"[EVAL] Exception during evaluation: {e}")
        # return a very poor score to avoid selecting these params
        return {"mean_amp": 1e6, "total_energy": 1e6, "reward": -1e9, "error": str(e)}


# -------------------------
# CLI / main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, default="lstm_state_estimator_fixed_predictions.csv", help="Predictions CSV path")
    p.add_argument("--iters", type=int, default=40, help="CEM iterations")
    p.add_argument("--batch", type=int, default=24, help="Candidates per iteration")
    p.add_argument("--elite_frac", type=float, default=0.2, help="Elite fraction")
    p.add_argument("--out_json", type=str, default="cem_best_params.json", help="Save best params JSON")
    p.add_argument("--init_B", type=float, default=-0.05)
    p.add_argument("--init_lambda_u", type=float, default=1e-2)
    p.add_argument("--init_lambda_du", type=float, default=1e-2)
    p.add_argument("--init_std_scale", type=float, default=0.5, help="Initial std multiplier for sampling")
    p.add_argument("--horizon", type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions}")

    # parameter vector = [B, lambda_u, lambda_du]
    param_names = ["B", "lambda_u", "lambda_du"]
    init_mean = np.array([args.init_B, args.init_lambda_u, args.init_lambda_du], dtype=float)
    init_std = np.abs(init_mean) * args.init_std_scale + args.init_std_scale  # ensure not too small

    print("[CEM] Starting optimization")
    print(f"[CEM] initial mean: {init_mean}, std: {init_std}")

    # Define evaluate function closure
    def eval_fn(pred_csv, B, lambda_u, lambda_du):
        # Enforce bounds: lambda >= 0, B reasonable range [-2,2]
        B = float(np.clip(B, -2.0, 2.0))
        lambda_u = float(max(lambda_u, 0.0))
        lambda_du = float(max(lambda_du, 0.0))
        return evaluate_wrapper(pred_csv, B, lambda_u, lambda_du)

    best, history = cem_optimize(evaluate_fn=eval_fn,
                                 pred_csv=args.predictions,
                                 param_names=param_names,
                                 init_mean=init_mean,
                                 init_std=init_std,
                                 n_iters=args.iters,
                                 batch_size=args.batch,
                                 elite_frac=args.elite_frac,
                                 save_prefix="cem_tune")

    print("[CEM] Optimization finished.")
    print("[CEM] Best score:", best["score"])
    print("[CEM] Best params:", best["params"])
    # Save best params
    best_params = {name: float(val) for name, val in zip(param_names, best["params"])}
    with open(args.out_json, "w") as f:
        json.dump({"best_score": best["score"], "best_params": best_params}, f, indent=2)
    print(f"[CEM] Saved best params to {args.out_json}")

    # If we have raw output from mpc evaluation, optionally save simulation results
    if "raw" in best.get("detail", {}):
        raw = best["detail"]["raw"]
        if isinstance(raw, dict):
            try:
                # if raw contains arrays, attempt to save summary
                pd.DataFrame(raw).to_csv("cem_best_raw_simulation.csv", index=False)
                print("[CEM] Saved raw best simulation to cem_best_raw_simulation.csv")
            except Exception:
                pass

if __name__ == "__main__":
    main()
