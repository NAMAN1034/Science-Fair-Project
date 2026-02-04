#!/usr/bin/env python3
"""
mpc_controller.py

Phase-aware Model Predictive Controller (MPC) for tremor phase modulation (simulation-only).

Usage:
    python mpc_controller.py --predictions my_model_predictions.csv --out mpc_results.csv

What this does (concise):
- Loads a predictions CSV (time_s, amp_pred, phase_pred) produced by your state estimator.
- Unwraps phase and treats instantaneous phase as the key state to control.
- Uses a simple, defensible linearized phase dynamics model:
      phi_{k+1} = phi_k + omega*dt + B * u_k
  where:
    - omega is estimated instantaneous angular velocity (from predicted phase)
    - u_k is the control action (stimulation "phase shift" command, abstract unit)
    - B is a small constant mapping control to phase shift per step (tunable)
- Sets up a receding-horizon quadratic program (QP) to choose a short sequence u_{0..Np-1} that
  minimizes phase tracking error to a desired offset (e.g., phi_target = phi + pi) while penalizing
  control energy and control-rate changes.
- Solves the QP with cvxpy (OSQP if available) at each step, applies only the first control action,
  simulates its effect on the phase and amplitude (toy model), and moves forward.
- Saves a CSV with the open-loop predicted phase/amplitude, applied control u, and simulated closed-loop results.

Important safety note:
- This is a **simulation-only** MPC meant to be used offline on logged data.
- Do NOT use the outputs of this script to apply electrical stimulation to humans without
  clinical oversight, safety hardware, and proper approvals.

Dependencies:
    numpy, pandas, scipy, cvxpy, osqp (optional), matplotlib (optional for quick plotting)

Author: Generated for ISEF workflow
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Try to import cvxpy and preferred solver
try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

# -----------------------
# Helper utilities
# -----------------------
def load_predictions(path: str):
    """
    Expects CSV with columns that include at least:
      - time_s   (seconds)
      - amp_pred (predicted IMU amplitude / tremor envelope)
      - phase_pred (predicted instantaneous phase in radians, wrapped or unwrapped)
    If columns are named differently, adapt reading logic here.
    """
    df = pd.read_csv(path)
    # accept multiple possible column names
    if "time_s" not in df.columns:
        # try common alternatives
        for c in df.columns:
            if "time" in c:
                df = df.rename(columns={c: "time_s"})
                break
    if "phase_pred" not in df.columns:
        # try other names
        for c in df.columns:
            if "phase" in c and "pred" in c:
                df = df.rename(columns={c: "phase_pred"})
                break
    if "amp_pred" not in df.columns:
        for c in df.columns:
            if "amp" in c and "pred" in c:
                df = df.rename(columns={c: "amp_pred"})
                break
    required = ["time_s", "phase_pred", "amp_pred"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Predictions file missing required column '{r}'. Columns found: {list(df.columns)}")
    return df

def unwrap_and_filter_phase(phase_wrapped: np.ndarray, window: int = 51, polyorder: int = 3):
    """
    Unwrap phase and apply mild smoothing to remove numerical jitter.
    Savitzky-Golay filter used for smoothing (window must be odd).
    """
    phase_unwrapped = np.unwrap(phase_wrapped)
    # Smoothing safe-guard
    if window >= len(phase_unwrapped):
        window = max(3, (len(phase_unwrapped) // 2) // 2 * 2 + 1)  # make odd
    try:
        phase_smooth = savgol_filter(phase_unwrapped, window_length=window if window>2 else 3, polyorder=polyorder)
    except Exception:
        phase_smooth = phase_unwrapped
    return phase_unwrapped, phase_smooth

# -----------------------
# MPC formulation
# -----------------------
def solve_mpc_phase(phi0, omega, dt, phi_ref_traj, Np,
                    B= -0.05,    # control-to-phase-effect coefficient (rad per unit u per step) - tunable
                    u_min=-1.0, u_max=1.0,
                    lambda_u=1e-2, lambda_du=1e-2,
                    warm_start: np.ndarray | None = None):
    """
    Solve finite-horizon QP:
      minimize sum_{k=0..Np-1} (phi_k - phi_ref_k)^2 + lambda_u * sum u_k^2 + lambda_du * sum (u_k - u_{k-1})^2
      s.t. phi_{k+1} = phi_k + omega*dt + B * u_k
           u_min <= u_k <= u_max

    Returns: optimal control sequence u_opt (Np,), and predicted phi sequence (Np+1,)
    Uses cvxpy if available, otherwise uses analytic QP solve (dense) via numpy for small horizons.
    """
    # Build dynamics matrices for linear discrete-time system:
    # phi = S * phi0 + T * u + c  (we will build explicit expression)
    # But easiest in cvxpy: variables phi[0..Np] and u[0..Np-1] with linear constraints.

    if HAS_CVXPY:
        phi = cp.Variable(Np+1)
        u = cp.Variable(Np)

        constraints = []
        constraints += [phi[0] == phi0]
        for k in range(Np):
            constraints += [phi[k+1] == phi[k] + omega*dt + B * u[k]]
            constraints += [u[k] >= u_min, u[k] <= u_max]

        # cost components
        cost = 0
        for k in range(Np):
            # reference at step k+1 maybe or k; choose k+1 to drive phi_{k+1} to ref_{k+1}
            cost += cp.square(phi[k+1] - phi_ref_traj[k+1])
            cost += lambda_u * cp.square(u[k])
            if k >= 1:
                cost += lambda_du * cp.square(u[k] - u[k-1])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        # solve
        try:
            # prefer OSQP for QP
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # fallback: return zeros
            return np.zeros(Np), np.full(Np+1, phi0 + omega * dt * np.arange(Np+1))
        u_opt = np.array(u.value).reshape(-1)
        phi_pred = np.array(phi.value).reshape(-1)
        return u_opt, phi_pred
    else:
        # Analytic QP solve fallback (dense formulation). Build quadratic in u only.
        # phi_{k+1} = phi0 + (k+1)*omega*dt + B * sum_{j=0..k} u_j
        # Let A be (Np x Np) lower-triangular ones, (A[k,j]=1 if j<=k else 0)
        # phi_vec = phi0 + (1..Np)*omega*dt + B * A @ u
        # Cost = ||phi_vec - phi_ref_vec||^2 + lambda_u * u^T u + lambda_du * ||D u||^2
        N = Np
        A = np.tril(np.ones((N, N)))
        c0 = phi0 + np.arange(1, N+1) * (omega*dt)
        phi_ref_vec = np.array([phi_ref_traj[k+1] for k in range(N)])  # targets for phi_{k+1}

        # Quadratic in u: (B A u + c0 - phi_ref)'(B A u + c0 - phi_ref) + lambda_u u'u + lambda_du (D u)'(D u)
        H = (B**2) * (A.T @ A) + lambda_u * np.eye(N)
        # add lambda_du for difference penalty -> D: (N-1)xN matrix with [ -1 1 ] rows
        D = np.zeros((N-1, N))
        for i in range(N-1):
            D[i, i] = -1.0
            D[i, i+1] = 1.0
        H = H + lambda_du * (D.T @ D)
        g = (B * A).T @ (c0 - phi_ref_vec)  # linear term is 2 * H u + 2 g in standard form; we'll solve directly via H u = -g
        # Solve (H) u = -g with box constraints via simple projected gradient descent (fallback)
        try:
            u = np.zeros(N)
            # projected gradient descent
            L = np.linalg.eigvalsh(H).max() if N>0 else 1.0
            step = 1.0 / (L + 1e-6)
            for it in range(200):
                grad = (H @ u + g)  # derivative of 0.5 u^T H u + g^T u
                u = u - step * grad
                # projection
                u = np.minimum(np.maximum(u, u_min), u_max)
            # compute phi_pred
            phi_pred = np.empty(N+1)
            phi_pred[0] = phi0
            for k in range(N):
                phi_pred[k+1] = phi_pred[k] + omega*dt + B * u[k]
            return u, phi_pred
        except Exception:
            return np.zeros(N), np.full(N+1, phi0 + omega * dt * np.arange(N+1))


# -----------------------
# Top-level MPC execution
# -----------------------
def run_receding_mpc(pred_csv: str,
                     out_csv: str = "mpc_results.csv",
                     horizon_seconds: float = 1.0,
                     dt_override: float | None = None,
                     B: float = -0.05,
                     u_min: float = -1.0,
                     u_max: float = 1.0,
                     lambda_u: float = 1e-2,
                     lambda_du: float = 1e-2,
                     desired_phase_offset: float = math.pi,
                     simulate_amp_effect: bool = True,
                     amp_effect_gain: float = 0.5):
    """
    pred_csv: input predictions file with columns time_s, amp_pred, phase_pred
    horizon_seconds: prediction horizon length used by MPC (e.g., 0.5 - 1.0 s)
    dt_override: if provided, override dt computed from file timestamps
    B: control-to-phase gain (rad per unit u per step); negative will advance phase in negative direction
    desired_phase_offset: the phase offset to target relative to current phase (e.g., pi means aim for opposite phase)
    simulate_amp_effect: if True, apply a simple multiplicative amplitude reduction when control is applied
    amp_effect_gain: scales how much amplitude reduces when control is nonzero (toy parameter)
    """
    df = load_predictions(pred_csv)
    t = df["time_s"].values.astype(float)
    phi = df["phase_pred"].values.astype(float)
    amp = df["amp_pred"].values.astype(float)

    # Unwrap and smooth phase
    phi_unwrapped = np.unwrap(phi)
    # smooth to reduce jitter (optional)
    try:
        from scipy.signal import savgol_filter
        win = min(101, len(phi_unwrapped)//2*2+1)
        if win >= 5:
            phi_smooth = savgol_filter(phi_unwrapped, window_length=win, polyorder=3)
        else:
            phi_smooth = phi_unwrapped
    except Exception:
        phi_smooth = phi_unwrapped

    # compute dt
    dt_arr = np.diff(t)
    dt = np.median(dt_arr) if dt_override is None else float(dt_override)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid dt; check time column in predictions CSV")
    print(f"[MPC] Using dt = {dt:.6f} s")

    N_total = len(t)
    Np = max(1, int(round(horizon_seconds / dt)))
    print(f"[MPC] Horizon steps Np = {Np} (horizon_seconds={horizon_seconds})")

    # Pre-allocate result arrays
    phi_no_control = phi_smooth.copy()
    amp_no_control = amp.copy()

    phi_sim = phi_smooth.copy()   # will be progressed by applied control in simulation
    amp_sim = amp.copy()

    applied_u = np.zeros(N_total)
    predicted_phi = np.zeros((N_total, Np+1))
    predicted_u_sequence = np.zeros((N_total, Np))

    # estimate instantaneous omega (phase velocity) via derivative of phi_smooth
    # Use central differences
    omega = np.gradient(phi_smooth, dt)  # rad/s

    # Receding horizon loop
    for i in range(0, N_total - Np):
        phi0 = phi_sim[i]
        # build reference trajectory: aim to shift phase by desired_phase_offset relative to open-loop predicted phase
        # Use open-loop predicted phi for future (from phi_no_control)
        phi_future_open = phi_no_control[i:i+Np+1]  # length Np+1
        # construct target as phi_future_open + desired_offset
        phi_ref = phi_future_open + desired_phase_offset

        # use local omega estimate; for stability use mean over horizon
        omega_local = float(np.mean(omega[i:i+Np]))

        # solve MPC for this window
        u_opt, phi_pred = solve_mpc_phase(phi0=phi0,
                                          omega=omega_local,
                                          dt=dt,
                                          phi_ref_traj=phi_ref,
                                          Np=Np,
                                          B=B,
                                          u_min=u_min,
                                          u_max=u_max,
                                          lambda_u=lambda_u,
                                          lambda_du=lambda_du,
                                          warm_start=None)
        # apply first control action
        u_apply = float(u_opt[0])
        applied_u[i] = u_apply
        predicted_phi[i, :] = phi_pred
        predicted_u_sequence[i, :] = u_opt

        # simulate effect on phase: we assume phi_sim evolves by one step of dynamics with u applied
        phi_sim[i+1] = phi_sim[i] + omega_local * dt + B * u_apply

        # simulate amplitude effect (toy): reduce amplitude immediately in proportion to |u|
        if simulate_amp_effect and amp_sim[i] > 0:
            # suppression factor between 1 and (1 - amp_effect_gain)
            supp = max(0.0, 1.0 - amp_effect_gain * abs(u_apply))
            amp_sim[i+1:] = amp_sim[i+1:] * supp  # crude persistent suppression
        # else: no amplitude change

    # Save results
    out_df = pd.DataFrame({
        "time_s": t,
        "amp_no_control": amp_no_control,
        "phi_no_control": phi_no_control,
        "amp_sim": amp_sim,
        "phi_sim": phi_sim,
        "u_applied": applied_u
    })
    # phi_sim is an array; ensure if phi_sim is 2D? It's 1D; predicted_phi is 2D omitted for brevity
    out_df.to_csv(out_csv, index=False)
    print(f"[MPC] Results saved to {out_csv}")

    # Optionally return arrays for further plotting
    return {
        "t": t,
        "amp_no_control": amp_no_control,
        "phi_no_control": phi_no_control,
        "amp_sim": amp_sim,
        "phi_sim": phi_sim,
        "u_applied": applied_u,
        "predicted_phi": predicted_phi,
        "predicted_u": predicted_u_sequence
    }


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, default="lstm_state_estimator_fixed_predictions.csv", help="Predictions CSV (time_s, amp_pred, phase_pred)")
    p.add_argument("--out", type=str, default="mpc_results.csv", help="Output CSV for MPC results")
    p.add_argument("--horizon", type=float, default=0.8, help="MPC horizon in seconds")
    p.add_argument("--dt", type=float, default=None, help="Override dt (seconds). If not set, read from file.")
    p.add_argument("--B", type=float, default=-0.05, help="Control-to-phase gain (rad per unit u per step)")
    p.add_argument("--u_min", type=float, default=-1.0)
    p.add_argument("--u_max", type=float, default=1.0)
    p.add_argument("--lambda_u", type=float, default=1e-2)
    p.add_argument("--lambda_du", type=float, default=1e-2)
    p.add_argument("--desired_offset", type=float, default=math.pi, help="Desired phase offset (radians), e.g., pi for destructive)")
    p.add_argument("--amp_effect_gain", type=float, default=0.5, help="Toy amplitude suppression gain for simulation")
    return p.parse_args()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    if not HAS_CVXPY:
        print("Warning: cvxpy not available. Falling back to dense PGD QP solver (may be slower / less robust). Install cvxpy+osqp for faster QP solves.")

    res = run_receding_mpc(pred_csv=args.predictions,
                           out_csv=args.out,
                           horizon_seconds=args.horizon,
                           dt_override=args.dt,
                           B=args.B,
                           u_min=args.u_min,
                           u_max=args.u_max,
                           lambda_u=args.lambda_u,
                           lambda_du=args.lambda_du,
                           desired_phase_offset=args.desired_offset,
                           simulate_amp_effect=True,
                           amp_effect_gain=args.amp_effect_gain)

    print("MPC simulation complete.")
    # Quick summary statistics
    applied = res["u_applied"]
    print(f"Applied controls: mean {np.mean(applied):.4f}, std {np.std(applied):.4f}, nonzero count {np.count_nonzero(applied)}")
    final_amp_reduction = res["amp_sim"][-1] / res["amp_no_control"][0] if res["amp_no_control"][0]>0 else 0.0
    print(f"Final amplitude reduction (simulated): {final_amp_reduction:.4f}")
    # End of mpc_controller.py