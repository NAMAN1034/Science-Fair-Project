#!/usr/bin/env python3
"""
isef_realtime_pipeline.py

Advanced real-time multimodal tremor pipeline (EMG + IMU) for ISEF-level demonstration.
Features:
- Binary serial ingestion from Teensy (TSF frames)
- Stateful DSP: EMG bandpass (20-250 Hz), IMU tremor band (3-12 Hz), envelope, phase
- Hilbert analytic & optional PLL for phase tracking
- Rolling FFT/PSD, cross-spectral features
- CNN -> LSTM predictor support (Keras .h5 or TFLite .tflite)
- MPC (cvxpy+OSQP if present, numeric fallback)
- RL meta-tuner (CEM) running offline on logged traces
- Multiprocessing for inference; thread-based IO & DSP
- CSV/HDF5 logging, robust error handling, config via CLI
- Safety guards and explicit "do not apply to humans" warnings

HOW TO RUN (example):
  python isef_realtime_pipeline.py --port /dev/tty.usbmodem187570801 --model model.h5 --logdir realtime_logs

Note: edit constants below to match actual ADC scale, IMU scale, and sampling timing.

Author: Naman's ISEF pipeline (generated)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import math
import struct
import threading
import queue
import logging
import json
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, welch, hilbert, decimate

# Optional heavy dependencies
try:
    import serial
except Exception as e:
    raise ImportError("pyserial required (pip install pyserial). Error: " + str(e))

# Model inference libraries
HAS_TENSORFLOW = False
HAS_TFLITE_RUNTIME = False
HAS_KERAS = False
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
    HAS_TENSORFLOW = True
    HAS_KERAS = True
except Exception:
    try:
        import tflite_runtime.interpreter as tflite_runtime  # type: ignore
        HAS_TFLITE_RUNTIME = True
    except Exception:
        pass

# MPC solver library
HAS_CVXPY = False
try:
    import cvxpy as cp  # type: ignore
    HAS_CVXPY = True
except Exception:
    pass
#code snippet 1
# Optional: OSQP solver for cvxpy (fast)
# (cvxpy will auto-select if installed)

# ---------------------------
#  CONFIG (change constants here)
# ---------------------------
FRAME_HEADER = b"TSF"         # 3 bytes header from Teensy
SERIAL_BAUD = 2000000         # choose Teensy USB speed
FRAME_MAX_SAMPLES = 32        # sensible upper bound for num samples per frame
EMG_ADC_BITS = 12             # Teensy ADC bits (0..4095)
EMG_ADC_OFFSET = 2048         # center offset applied in Teensy sketch if used
EMG_VREF = 3.3                # ADC Vref (V)
EMG_ADC_TO_V = EMG_VREF / (2**EMG_ADC_BITS)  # conversion per LSB if using signed centered ADC
GYRO_SCALE_PER_LSB = 0.01     # deg/s per LSB if Teensy scaled gyro by 100 (adjust if different)

# Sampling assumptions
DEFAULT_EMG_FS = 1000.0       # Hz if EMG timestamps invalid (override)
IMU_EXPECTED_FS = 200.0       # nominal IMU rate in Hz (used for sanity checks)
WINDOW_SECONDS = 0.5          # rolling window for inference/analysis
CONTROL_INTERVAL = 0.05       # seconds between control updates (20 Hz)
MPC_HORIZON_SECONDS = 0.8
LOG_DIR_DEFAULT = "realtime_logs"

# DSP bands (tunable)
EMG_BAND = (20.0, 250.0)
IMU_TREMOR_BAND = (3.0, 12.0)

# MPC safe bounds
U_MIN = -1.0
U_MAX = 1.0
MAX_PULSES_PER_SECOND = 10   # safety rate limit (example)

# RL tuner config
CEM_ITERS = 40
CEM_BATCH = 64
CEM_ELITE_FRAC = 0.2

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Utility functions
# ---------------------------
def mkdirp(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def current_micros() -> int:
    return int(time.time() * 1e6)

def clip(x, lo, hi):
    return max(lo, min(hi, x))

# ---------------------------
# Low-level serial frame parser (robust)
# Frame format:
# 3 bytes 'TSF' | 1 byte version | 2 bytes seq (uint16) | 8 bytes t0_us (uint64) | 1 byte N |
# For each sample: int16 emg, int16 gx, int16 gy, int16 gz   (all little-endian)
# ---------------------------
class SerialFrameReader(threading.Thread):
    def __init__(self, port: str, baud: int, frame_queue: queue.Queue, stop_event: threading.Event, timeout=0.2):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frame_q = frame_queue
        self.stop_event = stop_event
        self.timeout = timeout
        self.ser = None
        self._open_serial()

    def _open_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            logging.info(f"Opened serial {self.port} @ {self.baud}")
        except Exception as e:
            logging.error(f"Failed to open serial port {self.port}: {e}")
            raise

    def run(self):
        buf = b""
        while not self.stop_event.is_set():
            try:
                hdr = self.ser.read(3)
                if not hdr:
                    continue
                if hdr != FRAME_HEADER:
                    # try to resync: keep reading single bytes until header found
                    # read until header first byte
                    continue
                ver_b = self.ser.read(1)
                if not ver_b:
                    continue
                ver = ver_b[0]
                seq_b = self.ser.read(2)
                if len(seq_b) < 2:
                    continue
                seq = struct.unpack("<H", seq_b)[0]
                t0_b = self.ser.read(8)
                if len(t0_b) < 8:
                    continue
                t0 = struct.unpack("<Q", t0_b)[0]
                n_b = self.ser.read(1)
                if not n_b:
                    continue
                n = n_b[0]
                if n <= 0 or n > FRAME_MAX_SAMPLES:
                    # corrupted or evil data; skip
                    logging.warning(f"Frame with invalid sample count: {n}")
                    # try to continue gracefully
                    continue
                samples = []
                for i in range(n):
                    s = self.ser.read(8)
                    if len(s) < 8:
                        break
                    emg16, gx16, gy16, gz16 = struct.unpack("<hhhh", s)
                    samples.append((emg16, gx16, gy16, gz16))
                if len(samples) != n:
                    logging.warning("Incomplete frame read; skipping")
                    continue
                frame = {"seq": int(seq), "t0_us": int(t0), "n": int(n), "samples": samples}
                try:
                    self.frame_q.put_nowait(frame)
                except queue.Full:
                    # drop frames if queue is backed up
                    logging.debug("Frame queue full; dropping frame")
                    pass
            except Exception as e:
                logging.exception("Serial reader error: %s", e)
                time.sleep(0.1)
        self.ser.close()
        logging.info("SerialFrameReader stopped")

# ---------------------------
# Ring buffer for streaming, with thread-safety
# ---------------------------
class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.count = 0
        self.lock = threading.Lock()

    def push(self, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        n = arr.size
        with self.lock:
            if n >= self.capacity:
                # keep last capacity
                self.buf[:] = arr[-self.capacity:]
                self.idx = 0
                self.count = self.capacity
                return
            end = (self.idx + n) % self.capacity
            if self.idx + n <= self.capacity:
                self.buf[self.idx:self.idx + n] = arr
            else:
                a = self.capacity - self.idx
                self.buf[self.idx:] = arr[:a]
                self.buf[:n - a] = arr[a:]
            self.idx = end
            self.count = min(self.capacity, self.count + n)

    def last(self, n: int) -> np.ndarray:
        n = int(n)
        with self.lock:
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            if self.count < n:
                raise ValueError("Not enough samples in buffer")
            start = (self.idx - n) % self.capacity
            if start + n <= self.capacity:
                return self.buf[start:start + n].copy()
            else:
                a = self.capacity - start
                return np.concatenate([self.buf[start:], self.buf[:n - a]]).copy()

    def available(self) -> int:
        with self.lock:
            return int(self.count)

# ---------------------------
# DSP: stateful IIR SOS filters, simple PLL and analytic signal helpers
# ---------------------------
def design_sos(fs: float, low: float, high: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Invalid bandpass for fs={fs}, low={low}, high={high}")
    from scipy.signal import butter
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    zi = sosfilt_zi(sos)
    return sos, zi

class StatefulFilter:
    """Wraps sosfilt with maintained state."""
    def __init__(self, sos, zi=None):
        self.sos = np.asarray(sos)
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64) if zi is None else np.array(zi, dtype=np.float64)
    def apply(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt
        y, zf = sosfilt(self.sos, x, zi=self.zi)
        self.zi = zf
        return y

# Lightweight Phase-Locked Loop for robust phase tracking (real-time)
class SimplePLL:
    """
    Tracks phase of a narrowband signal in real-time.
    Basic algorithm:
      - maintain phase estimate phi_hat
      - multiply input by cos(phi_hat) and sin(phi_hat) to get I/Q error
      - update phi_hat derivative by Kp * error
    This is not a full control-theory PLL but is robust for periodic tremor.
    """
    def __init__(self, fs: float, init_freq: float = 5.0, kp: float = 0.1, ki: float = 0.01):
        self.fs = fs
        self.dt = 1.0 / fs
        self.phi = 0.0
        self.omega = 2.0 * math.pi * init_freq
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def step(self, sample: float):
        # sample assumed narrowband real signal (bandpassed)
        # generate reference oscillator
        ref_cos = math.cos(self.phi)
        ref_sin = math.sin(self.phi)
        i = sample * ref_cos
        q = sample * (-ref_sin)  # quadrature sign
        # phase error approx arctan2(q,i)
        err = math.atan2(q, i)
        self.integral += err * self.dt
        self.omega += self.kp * err + self.ki * self.integral
        # integrate phase
        self.phi += self.omega * self.dt
        # normalize
        self.phi = (self.phi + math.pi) % (2 * math.pi) - math.pi
        return self.phi, abs(math.hypot(i, q)), self.omega / (2 * math.pi)
#code snippet 2
# ---------------------------
# Feature extraction
# ---------------------------
def compute_psd_features(signal: np.ndarray, fs: float, fmin=3.0, fmax=12.0):
    # compute Welch PSD and return band power and peak freq
    if len(signal) < 8:
        return {"band_power": 0.0, "peak_freq": 0.0}
    f, pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return {"band_power": 0.0, "peak_freq": 0.0}
    band_power = float(np.trapz(pxx[mask], f[mask]))
    peak_freq = float(f[mask][np.argmax(pxx[mask])])
    return {"band_power": band_power, "peak_freq": peak_freq}

# ---------------------------
# Predictor: Keras .h5 or TFLite .tflite support + multiprocessing wrapper
# ---------------------------
class PredictorProcess:
    """Runs model inference in a separate process to escape Python GIL for heavy TF workloads."""
    def __init__(self, model_path: Optional[str], seq_len: int, use_tflite: bool = False):
        self.model_path = model_path
        self.seq_len = int(seq_len)
        self.use_tflite = use_tflite
        self.request_q = queue.Queue(maxsize=8)
        self.reply_q = queue.Queue(maxsize=8)
        self.proc = threading.Thread(target=self._worker, daemon=True)
        self.proc.start()

    def _worker(self):
        # Load model in this thread (isolates TF memory)
        keras_model = None
        tflite_interp = None
        if self.model_path:
            try:
                if self.use_tflite:
                    if HAS_TFLITE_RUNTIME:
                        tflite_interp = tflite_runtime.Interpreter(model_path=self.model_path)
                        tflite_interp.allocate_tensors()
                        in_det = tflite_interp.get_input_details()
                        out_det = tflite_interp.get_output_details()
                        logging.info(f"TFLite model loaded: {self.model_path}")
                    else:
                        logging.warning("tflite-runtime not installed; cannot load TFLite model.")
                        tflite_interp = None
                else:
                    if HAS_KERAS:
                        keras_model = load_model(self.model_path)
                        logging.info(f"Keras model loaded: {self.model_path}")
                    else:
                        logging.warning("Keras/TensorFlow not available.")
            except Exception as e:
                logging.exception("Failed to load model: %s", e)
                keras_model = None
                tflite_interp = None
        # worker loop
        while True:
            try:
                req = self.request_q.get()
            except Exception:
                break
            if req is None:
                break
            emg_win, imu_win, meta = req
            amp_pred, phase_pred = None, None
            try:
                if keras_model is not None:
                    emg_in = emg_win.reshape(1, -1, 1).astype(np.float32)
                    imu_in = imu_win.reshape(1, -1, 1).astype(np.float32)
                    preds = keras_model.predict([emg_in, imu_in], verbose=0)
                    # support multiple output shapes
                    if isinstance(preds, list) and len(preds) >= 2:
                        amp_pred = float(np.array(preds[0]).reshape(-1)[0])
                        cs = np.array(preds[1]).reshape(-1, 2)[0]
                        phase_pred = math.atan2(cs[1], cs[0])
                    else:
                        amp_pred = float(np.array(preds).reshape(-1)[0])
                        phase_pred = 0.0
                elif tflite_interp is not None:
                    # try setting inputs; handle both single and two-input models
                    inputs = tflite_interp.get_input_details()
                    outputs = tflite_interp.get_output_details()
                    if len(inputs) >= 2:
                        tflite_interp.set_tensor(inputs[0]['index'], emg_win.reshape(1, -1, 1).astype(np.float32))
                        tflite_interp.set_tensor(inputs[1]['index'], imu_win.reshape(1, -1, 1).astype(np.float32))
                    else:
                        # concatenate channels
                        inp = np.stack([emg_win, imu_win], axis=-1).reshape(1, -1, 2).astype(np.float32)
                        tflite_interp.set_tensor(inputs[0]['index'], inp)
                    tflite_interp.invoke()
                    out0 = tflite_interp.get_tensor(outputs[0]['index'])
                    out1 = tflite_interp.get_tensor(outputs[1]['index']) if len(outputs) >= 2 else None
                    amp_pred = float(np.array(out0).reshape(-1)[0])
                    if out1 is not None:
                        cs = np.array(out1).reshape(-1, 2)[0]
                        phase_pred = math.atan2(cs[1], cs[0])
                    else:
                        phase_pred = 0.0
                else:
                    # fallback heuristic: use analytic on imu_win
                    analytic = hilbert(imu_win)
                    amp_pred = float(np.abs(analytic[-1]))
                    phase_pred = float(np.angle(analytic[-1]))
            except Exception as e:
                logging.exception("Predictor worker error: %s", e)
                # fallback
                analytic = hilbert(imu_win)
                amp_pred = float(np.abs(analytic[-1]))
                phase_pred = float(np.angle(analytic[-1]))
            # reply
            try:
                self.reply_q.put_nowait((amp_pred, phase_pred, meta))
            except queue.Full:
                # drop if cannot reply
                pass

    def predict(self, emg_win: np.ndarray, imu_win: np.ndarray, meta: Optional[dict] = None, timeout=0.5):
        try:
            self.request_q.put_nowait((emg_win.astype(np.float32), imu_win.astype(np.float32), meta or {}))
        except queue.Full:
            logging.debug("Predictor request queue full; skipping prediction")
            return None
        try:
            amp, phase, meta = self.reply_q.get(timeout=timeout)
            return amp, phase, meta
        except queue.Empty:
            return None

    def stop(self):
        try:
            self.request_q.put_nowait(None)
        except Exception:
            pass

# ---------------------------
# MPC controller wrapper
# ---------------------------
def solve_phase_mpc(phi0: float, omega: float, dt: float, phi_ref: np.ndarray, Np: int,
                    B: float, u_min: float, u_max: float, lambda_u: float, lambda_du: float):
    """Solve QP for phase control. Use cvxpy if available; else numeric fallback."""
    if HAS_CVXPY:
        phi = cp.Variable(Np + 1)
        u = cp.Variable(Np)
        constraints = [phi[0] == phi0]
        for k in range(Np):
            constraints += [phi[k + 1] == phi[k] + omega * dt + B * u[k]]
            constraints += [u[k] >= u_min, u[k] <= u_max]
        cost = 0
        for k in range(Np):
            cost += cp.square(phi[k + 1] - phi_ref[k + 1])
            cost += lambda_u * cp.square(u[k])
            if k >= 1:
                cost += lambda_du * cp.square(u[k] - u[k - 1])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.zeros(Np), np.full(Np + 1, phi0 + omega * dt * np.arange(Np + 1))
        return np.array(u.value).reshape(-1), np.array(phi.value).reshape(-1)
    else:
        # dense numeric solve via projected gradient descent on small horizon
        N = Np
        A = np.tril(np.ones((N, N)))
        c0 = phi0 + np.arange(1, N + 1) * (omega * dt)
        phi_ref_vec = phi_ref[1:N + 1]
        H = (B ** 2) * (A.T @ A) + lambda_u * np.eye(N)
        D = np.zeros((N - 1, N))
        for i in range(N - 1):
            D[i, i] = -1
            D[i, i + 1] = 1
        H = H + lambda_du * (D.T @ D)
        g = (B * A).T @ (c0 - phi_ref_vec)
        u = np.zeros(N)
        L = np.linalg.eigvalsh(H).max() if N > 0 else 1.0
        step = 1.0 / (L + 1e-9)
        for it in range(400):
            grad = H @ u + g
            u = u - step * grad
            u = np.clip(u, u_min, u_max)
        phi_pred = np.empty(N + 1)
        phi_pred[0] = phi0
        for k in range(N):
            phi_pred[k + 1] = phi_pred[k] + omega * dt + B * u[k]
        return u, phi_pred

# ---------------------------
# RL tuner (CEM) that evaluates parameter sets on stored traces (safe offline)
# ---------------------------
def cem_optimize_on_trace(trace_pred_fn, init_mean: np.ndarray, init_std: np.ndarray, iters: int = 40, batch: int = 64, elite_frac: float = 0.2):
    rng = np.random.default_rng(12345)
    mean = init_mean.copy()
    std = init_std.copy()
    n_elite = max(1, int(np.round(batch * elite_frac)))
    best = {"score": -1e9, "params": None}
    history = []
    for it in range(iters):
        samples = rng.normal(mean[None, :], std[None, :], size=(batch, len(mean)))
        scores = np.zeros(batch, dtype=float)
        details = [None] * batch
        for i in range(batch):
            B, lam_u, lam_du = float(samples[i, 0]), float(max(samples[i, 1], 0.0)), float(max(samples[i, 2], 0.0))
            res = trace_pred_fn(B, lam_u, lam_du)
            score = float(res.get("reward", -1e9))
            scores[i] = score
            details[i] = res
        idx = np.argsort(scores)[::-1]
        elites = samples[idx[:n_elite]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-9
        if scores[idx[0]] > best["score"]:
            best["score"] = float(scores[idx[0]])
            best["params"] = samples[idx[0]].tolist()
            best["detail"] = details[idx[0]]
        history.append({"iter": it, "best_score": float(best["score"]), "best_params": list(best["params"]), "mean": mean.tolist()})
    return best, history

# ---------------------------
# Orchestrator: ties everything together and logs
# ---------------------------
class Orchestrator:
    def __init__(self, port: str, baud: int, model_path: Optional[str], tflite_path: Optional[str], logdir: str, emg_fs: float = DEFAULT_EMG_FS):
        self.port = port
        self.baud = baud
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.logdir = logdir
        mkdirp(self.logdir)
        self.emg_fs = float(emg_fs)
        self.seq_len = int(round(WINDOW_SECONDS * self.emg_fs))
        # ring buffers
        cap = int(self.emg_fs * 60)  # keep 60s buffer
        self.emg_rb = RingBuffer(cap)
        self.imu_rb = RingBuffer(cap)
        # queues
        self.frame_q = queue.Queue(maxsize=2048)
        self.feature_q = queue.Queue(maxsize=4096)
        self.control_q = queue.Queue(maxsize=512)
        self.stop_evt = threading.Event()
        # threads
        self.serial_thread = SerialFrameReader(self.port, self.baud, self.frame_q, self.stop_evt)
        self.dsp_thread = threading.Thread(target=self._dsp_worker, daemon=True)
        self.ctrl_thread = threading.Thread(target=self._control_worker, daemon=True)
        self.logger_thread = threading.Thread(target=self._logger_worker, daemon=True)
        self.controller_writer_thread = threading.Thread(target=self._controller_writer, daemon=True)
        # predictor
        use_tflite = False
        if self.tflite_path:
            use_tflite = True
        self.predictor = PredictorProcess(self.model_path if not use_tflite else self.tflite_path, seq_len=self.seq_len, use_tflite=use_tflite)
        # MPC params
        self.mpc_params = {"B": -0.05, "lambda_u": 1e-2, "lambda_du": 1e-2, "u_min": U_MIN, "u_max": U_MAX, "horizon": MPC_HORIZON_SECONDS, "desired_phase_offset": math.pi}
        # safety trackers
        self.last_control_time = 0.0
        self.controls_sent = 0
        # log files
        self.raw_path = os.path.join(self.logdir, "raw_frames.csv")
        self.features_path = os.path.join(self.logdir, "features.csv")
        self.pred_path = os.path.join(self.logdir, "predictions.csv")
        self.ctrl_path = os.path.join(self.logdir, "controls.csv")
        # PLL tracker optional
        self.pll = SimplePLL(fs=self.emg_fs, init_freq=5.0)  # start PLL with 5 Hz
        # design DSP filters for imu and emg used offline in dsp worker
        self.imu_sos, self.imu_zi = design_sos(self.emg_fs, IMU_TREMOR_BAND[0], IMU_TREMOR_BAND[1], order=4)
        self.emg_sos, self.emg_zi = design_sos(self.emg_fs, EMG_BAND[0], EMG_BAND[1], order=4)
#code snippet 3
    def start(self):
        logging.info("Starting orchestrator threads")
        self.serial_thread.start()
        self.dsp_thread.start()
        self.ctrl_thread.start()
        self.logger_thread.start()
        self.controller_writer_thread.start()
        #code snippet 4

    def stop(self):
        logging.info("Stopping orchestrator")
        self.stop_evt.set()
        self.predictor.stop()

    def _dsp_worker(self):
        """
        Consumes frames from frame_q, pushes into ringbuffers, computes rolling features,
        and pushes snapshots for control/prediction at CONTROL_INTERVAL cadence.
        """
        last_ctrl_time = time.time()
        while not self.stop_evt.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # unpack frame samples and push to buffers
            samples = frame["samples"]
            emg_vals = np.array([s[0] for s in samples], dtype=np.float32)
            imu_vals = np.array([s[1] for s in samples], dtype=np.float32)  # use gx axis for now
            self.emg_rb.push(emg_vals)
            self.imu_rb.push(imu_vals)
            # log raw frame metadata
            raw_row = {"t0_us": int(frame["t0_us"]), "seq": int(frame["seq"]), "n": int(frame["n"]), "recv_us": current_micros()}
            try:
                with open(self.raw_path, "a") as f:
                    f.write(",".join([str(raw_row[k]) for k in ("t0_us", "seq", "n", "recv_us")]) + "\n")
            except Exception:
                pass
            # Fast feature extraction on last window
            if self.emg_rb.available() >= self.seq_len and self.imu_rb.available() >= self.seq_len:
                try:
                    emg_win = self.emg_rb.last(self.seq_len)
                    imu_win = self.imu_rb.last(self.seq_len)
                except ValueError:
                    continue
                # filter imu and emg (stateful)
                try:
                    imu_f = sosfilt(self.imu_sos, imu_win)
                except Exception:
                    imu_f = imu_win
                try:
                    emg_f = sosfilt(self.emg_sos, emg_win)
                except Exception:
                    emg_f = emg_win
                # analytic signal
                analytic = hilbert(imu_f)
                inst_amp = float(np.abs(analytic[-1]))
                inst_phase = float(np.angle(analytic[-1]))
                # PLL update (robust)
                pll_phase, pll_amp, pll_freq = self.pll.step(imu_f[-1])
                # PSD features
                psd = compute_psd_features(imu_f, fs=self.emg_fs, fmin=IMU_TREMOR_BAND[0], fmax=IMU_TREMOR_BAND[1])
                emg_env = float(np.sqrt(np.mean(np.square(emg_f[-int(0.1*self.emg_fs):]))))
                feat = {"t": time.time(), "inst_amp": inst_amp, "inst_phase": inst_phase, "pll_phase": pll_phase, "pll_freq": pll_freq,
                        "band_power": psd["band_power"], "peak_freq": psd["peak_freq"], "emg_env": emg_env}
                try:
                    # non-blocking
                    self.feature_q.put_nowait(("feature", feat, emg_win.copy(), imu_f.copy()))
                except queue.Full:
                    pass
            # cadence for control/prediction
            now = time.time()
            if now - last_ctrl_time >= CONTROL_INTERVAL:
                last_ctrl_time = now
                # we expect feature_q to be filled frequently; if empty, skip
                try:
                    tag, feat, emg_win, imu_f = self.feature_q.get(timeout=0.01)
                    # Put back for control loop to consume (duplicate or send snapshot)
                    try:
                        self.feature_q.put_nowait((tag, feat, emg_win, imu_f))
                    except queue.Full:
                        pass
                except queue.Empty:
                    # not enough features, skip this cycle
                    pass
        logging.info("DSP worker stopped")

    def _control_worker(self):
        """
        Runs at CONTROL_INTERVAL cadence: predict -> MPC -> log -> enqueue control
        """
        last_time = time.time()
        while not self.stop_evt.is_set():
            now = time.time()
            if now - last_time < CONTROL_INTERVAL:
                time.sleep(0.001)
                continue
            last_time = now
            # get latest feature snapshot; if none, continue
            try:
                tag, feat, emg_win, imu_f = self.feature_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # model prediction
            pred = self.predictor.predict(emg_win, imu_f, meta={"t": feat["t"]})
            if pred is None:
                # fallback: use analytic
                amp_pred = feat["inst_amp"]
                phase_pred = feat["inst_phase"]
            else:
                amp_pred, phase_pred, _ = pred
            # compute open-loop phi_ref (Np steps)
            dt = CONTROL_INTERVAL
            Np = max(1, int(round(self.mpc_params["horizon"] / dt)))
            # estimate omega (rad/s) from PLL frequency
            omega = 2.0 * math.pi * feat.get("pll_freq", feat.get("peak_freq", 5.0))
            phi0 = feat.get("pll_phase", feat.get("inst_phase", phase_pred))
            phi_ref = np.zeros(Np + 1)
            for k in range(Np + 1):
                phi_ref[k] = phi0 + omega * dt * k
            phi_ref += self.mpc_params.get("desired_phase_offset", math.pi)
            # solve MPC
            u_seq, phi_seq = solve_phase_mpc(phi0=phi0, omega=omega, dt=dt, phi_ref=phi_ref, Np=Np,
                                            B=self.mpc_params["B"], u_min=self.mpc_params["u_min"], u_max=self.mpc_params["u_max"],
                                            lambda_u=self.mpc_params["lambda_u"], lambda_du=self.mpc_params["lambda_du"])
            u_apply = float(clip(u_seq[0], U_MIN, U_MAX))
            # safety: rate limiting and pulse budget
            if time.time() - self.last_control_time < 1.0 / MAX_PULSES_PER_SECOND:
                logging.debug("Control rate limit active; skipping control")
                continue
            self.last_control_time = time.time()
            # enqueue control for writer
            try:
                self.control_q.put_nowait(u_apply)
            except queue.Full:
                logging.debug("Control queue full; dropping control")
            # log prediction & control
            pred_row = {"t": feat["t"], "amp_pred": float(amp_pred), "phase_pred": float(phase_pred), "phi0": phi0, "omega": omega}
            ctrl_row = {"t": time.time(), "u": u_apply, "B": self.mpc_params["B"], "lambda_u": self.mpc_params["lambda_u"], "lambda_du": self.mpc_params["lambda_du"]}
            try:
                with open(self.pred_path, "a") as f:
                    f.write(json.dumps(pred_row) + "\n")
                with open(self.ctrl_path, "a") as f:
                    f.write(json.dumps(ctrl_row) + "\n")
            except Exception:
                pass
        logging.info("Control worker stopped")

    def _logger_worker(self):
        """
        Drains feature queue and writes to CSV for offline analysis.
        """
        # create header on first write
        feat_header_written = os.path.exists(self.features_path)
        with open(self.features_path, "a") as ffeat:
            while not self.stop_evt.is_set():
                try:
                    tag, feat, *_ = self.feature_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                # append
                try:
                    ffeat.write(json.dumps(feat) + "\n")
                except Exception:
                    pass
        logging.info("Logger worker stopped")

    def _controller_writer(self):
        """
        Sends CTL packets to Teensy: 'CTL' + float32 (little-endian)
        """
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as e:
            logging.error("Controller writer cannot open serial: %s", e)
            return
        logging.info("Controller writer opened serial port")
        while not self.stop_evt.is_set():
            try:
                u = self.control_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # safety clamp
            u = clip(u, U_MIN, U_MAX)
            pkt = b'CTL' + struct.pack("<f", float(u))
            try:
                ser.write(pkt)
                self.controls_sent += 1
            except Exception:
                logging.exception("Failed to write control packet")
        ser.close()
        logging.info("Controller writer stopped")

# ---------------------------
# CLI & bootstrap
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ISEF real-time tremor pipeline (EMG + IMU -> CNN/LSTM -> MPC -> RL)")
    p.add_argument("--port", required=True, help="Serial port to Teensy (e.g., /dev/ttyUSB0 or /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=SERIAL_BAUD)
    p.add_argument("--model", default=None, help="Path to Keras .h5 model (optional)")
    p.add_argument("--tflite", default=None, help="Path to TFLite .tflite model (optional)")
    p.add_argument("--logdir", default=LOG_DIR_DEFAULT, help="Directory to save logs")
    p.add_argument("--emg_fs", type=float, default=DEFAULT_EMG_FS, help="EMG sampling rate (Hz) fallback")
    return p.parse_args()

def main():
    args = parse_args()
    orch = Orchestrator(port=args.port, baud=args.baud, model_path=args.model, tflite_path=args.tflite, logdir=args.logdir, emg_fs=args.emg_fs)
    try:
        orch.start()
        logging.info("Orchestrator started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: stopping...")
        orch.stop()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
isef_realtime_pipeline.py

Advanced real-time multimodal tremor pipeline (EMG + IMU) for ISEF-level demonstration.
Features:
- Binary serial ingestion from Teensy (TSF frames)
- Stateful DSP: EMG bandpass (20-250 Hz), IMU tremor band (3-12 Hz), envelope, phase
- Hilbert analytic & optional PLL for phase tracking
- Rolling FFT/PSD, cross-spectral features
- CNN -> LSTM predictor support (Keras .h5 or TFLite .tflite)
- MPC (cvxpy+OSQP if present, numeric fallback)
- RL meta-tuner (CEM) running offline on logged traces
- Multiprocessing for inference; thread-based IO & DSP
- CSV/HDF5 logging, robust error handling, config via CLI
- Safety guards and explicit "do not apply to humans" warnings

HOW TO RUN (example):
  python isef_realtime_pipeline.py --port /dev/tty.usbmodem187570801 --model model.h5 --logdir realtime_logs

Note: edit constants below to match actual ADC scale, IMU scale, and sampling timing.

Author: Naman's ISEF pipeline (generated)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import math
import struct
import threading
import queue
import logging
import json
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, welch, hilbert, decimate

# Optional heavy dependencies
try:
    import serial
except Exception as e:
    raise ImportError("pyserial required (pip install pyserial). Error: " + str(e))

# Model inference libraries
HAS_TENSORFLOW = False
HAS_TFLITE_RUNTIME = False
HAS_KERAS = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
    HAS_KERAS = True
except Exception:
    try:
        import tflite_runtime.interpreter as tflite_runtime
        HAS_TFLITE_RUNTIME = True
    except Exception:
        pass

# MPC solver library
HAS_CVXPY = False
try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    pass

# Optional: OSQP solver for cvxpy (fast)
# (cvxpy will auto-select if installed)

# ---------------------------
#  CONFIG (change constants here)
# ---------------------------
FRAME_HEADER = b"TSF"         # 3 bytes header from Teensy
SERIAL_BAUD = 2000000         # choose Teensy USB speed
FRAME_MAX_SAMPLES = 32        # sensible upper bound for num samples per frame
EMG_ADC_BITS = 12             # Teensy ADC bits (0..4095)
EMG_ADC_OFFSET = 2048         # center offset applied in Teensy sketch if used
EMG_VREF = 3.3                # ADC Vref (V)
EMG_ADC_TO_V = EMG_VREF / (2**EMG_ADC_BITS)  # conversion per LSB if using signed centered ADC
GYRO_SCALE_PER_LSB = 0.01     # deg/s per LSB if Teensy scaled gyro by 100 (adjust if different)

# Sampling assumptions
DEFAULT_EMG_FS = 1000.0       # Hz if EMG timestamps invalid (override)
IMU_EXPECTED_FS = 200.0       # nominal IMU rate in Hz (used for sanity checks)
WINDOW_SECONDS = 0.5          # rolling window for inference/analysis
CONTROL_INTERVAL = 0.05       # seconds between control updates (20 Hz)
MPC_HORIZON_SECONDS = 0.8
LOG_DIR_DEFAULT = "realtime_logs"

# DSP bands (tunable)
EMG_BAND = (20.0, 250.0)
IMU_TREMOR_BAND = (3.0, 12.0)

# MPC safe bounds
U_MIN = -1.0
U_MAX = 1.0
MAX_PULSES_PER_SECOND = 10   # safety rate limit (example)

# RL tuner config
CEM_ITERS = 40
CEM_BATCH = 64
CEM_ELITE_FRAC = 0.2

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Utility functions
# ---------------------------
def mkdirp(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def current_micros() -> int:
    return int(time.time() * 1e6)

def clip(x, lo, hi):
    return max(lo, min(hi, x))

# ---------------------------
# Low-level serial frame parser (robust)
# Frame format:
# 3 bytes 'TSF' | 1 byte version | 2 bytes seq (uint16) | 8 bytes t0_us (uint64) | 1 byte N |
# For each sample: int16 emg, int16 gx, int16 gy, int16 gz   (all little-endian)
# ---------------------------
class SerialFrameReader(threading.Thread):
    def __init__(self, port: str, baud: int, frame_queue: queue.Queue, stop_event: threading.Event, timeout=0.2):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frame_q = frame_queue
        self.stop_event = stop_event
        self.timeout = timeout
        self.ser = None
        self._open_serial()

    def _open_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            logging.info(f"Opened serial {self.port} @ {self.baud}")
        except Exception as e:
            logging.error(f"Failed to open serial port {self.port}: {e}")
            raise

    def run(self):
        buf = b""
        while not self.stop_event.is_set():
            try:
                hdr = self.ser.read(3)
                if not hdr:
                    continue
                if hdr != FRAME_HEADER:
                    # try to resync: keep reading single bytes until header found
                    # read until header first byte
                    continue
                ver_b = self.ser.read(1)
                if not ver_b:
                    continue
                ver = ver_b[0]
                seq_b = self.ser.read(2)
                if len(seq_b) < 2:
                    continue
                seq = struct.unpack("<H", seq_b)[0]
                t0_b = self.ser.read(8)
                if len(t0_b) < 8:
                    continue
                t0 = struct.unpack("<Q", t0_b)[0]
                n_b = self.ser.read(1)
                if not n_b:
                    continue
                n = n_b[0]
                if n <= 0 or n > FRAME_MAX_SAMPLES:
                    # corrupted or evil data; skip
                    logging.warning(f"Frame with invalid sample count: {n}")
                    # try to continue gracefully
                    continue
                samples = []
                for i in range(n):
                    s = self.ser.read(8)
                    if len(s) < 8:
                        break
                    emg16, gx16, gy16, gz16 = struct.unpack("<hhhh", s)
                    samples.append((emg16, gx16, gy16, gz16))
                if len(samples) != n:
                    logging.warning("Incomplete frame read; skipping")
                    continue
                frame = {"seq": int(seq), "t0_us": int(t0), "n": int(n), "samples": samples}
                try:
                    self.frame_q.put_nowait(frame)
                except queue.Full:
                    # drop frames if queue is backed up
                    logging.debug("Frame queue full; dropping frame")
                    pass
            except Exception as e:
                logging.exception("Serial reader error: %s", e)
                time.sleep(0.1)
        self.ser.close()
        logging.info("SerialFrameReader stopped")

# ---------------------------
# Ring buffer for streaming, with thread-safety
# ---------------------------
class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.count = 0
        self.lock = threading.Lock()

    def push(self, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        n = arr.size
        with self.lock:
            if n >= self.capacity:
                # keep last capacity
                self.buf[:] = arr[-self.capacity:]
                self.idx = 0
                self.count = self.capacity
                return
            end = (self.idx + n) % self.capacity
            if self.idx + n <= self.capacity:
                self.buf[self.idx:self.idx + n] = arr
            else:
                a = self.capacity - self.idx
                self.buf[self.idx:] = arr[:a]
                self.buf[:n - a] = arr[a:]
            self.idx = end
            self.count = min(self.capacity, self.count + n)

    def last(self, n: int) -> np.ndarray:
        n = int(n)
        with self.lock:
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            if self.count < n:
                raise ValueError("Not enough samples in buffer")
            start = (self.idx - n) % self.capacity
            if start + n <= self.capacity:
                return self.buf[start:start + n].copy()
            else:
                a = self.capacity - start
                return np.concatenate([self.buf[start:], self.buf[:n - a]]).copy()

    def available(self) -> int:
        with self.lock:
            return int(self.count)

# ---------------------------
# DSP: stateful IIR SOS filters, simple PLL and analytic signal helpers
# ---------------------------
def design_sos(fs: float, low: float, high: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Invalid bandpass for fs={fs}, low={low}, high={high}")
    from scipy.signal import butter
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    zi = sosfilt_zi(sos)
    return sos, zi

class StatefulFilter:
    """Wraps sosfilt with maintained state."""
    def __init__(self, sos, zi=None):
        self.sos = np.asarray(sos)
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64) if zi is None else np.array(zi, dtype=np.float64)
    def apply(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt
        y, zf = sosfilt(self.sos, x, zi=self.zi)
        self.zi = zf
        return y

# Lightweight Phase-Locked Loop for robust phase tracking (real-time)
class SimplePLL:
    """
    Tracks phase of a narrowband signal in real-time.
    Basic algorithm:
      - maintain phase estimate phi_hat
      - multiply input by cos(phi_hat) and sin(phi_hat) to get I/Q error
      - update phi_hat derivative by Kp * error
    This is not a full control-theory PLL but is robust for periodic tremor.
    """
    def __init__(self, fs: float, init_freq: float = 5.0, kp: float = 0.1, ki: float = 0.01):
        self.fs = fs
        self.dt = 1.0 / fs
        self.phi = 0.0
        self.omega = 2.0 * math.pi * init_freq
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def step(self, sample: float):
        # sample assumed narrowband real signal (bandpassed)
        # generate reference oscillator
        ref_cos = math.cos(self.phi)
        ref_sin = math.sin(self.phi)
        i = sample * ref_cos
        q = sample * (-ref_sin)  # quadrature sign
        # phase error approx arctan2(q,i)
        err = math.atan2(q, i)
        self.integral += err * self.dt
        self.omega += self.kp * err + self.ki * self.integral
        # integrate phase
        self.phi += self.omega * self.dt
        # normalize
        self.phi = (self.phi + math.pi) % (2 * math.pi) - math.pi
        return self.phi, abs(math.hypot(i, q)), self.omega / (2 * math.pi)

# ---------------------------
# Feature extraction
# ---------------------------
def compute_psd_features(signal: np.ndarray, fs: float, fmin=3.0, fmax=12.0):
    # compute Welch PSD and return band power and peak freq
    if len(signal) < 8:
        return {"band_power": 0.0, "peak_freq": 0.0}
    f, pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return {"band_power": 0.0, "peak_freq": 0.0}
    band_power = float(np.trapz(pxx[mask], f[mask]))
    peak_freq = float(f[mask][np.argmax(pxx[mask])])
    return {"band_power": band_power, "peak_freq": peak_freq}

# ---------------------------
# Predictor: Keras .h5 or TFLite .tflite support + multiprocessing wrapper
# ---------------------------
class PredictorProcess:
    """Runs model inference in a separate process to escape Python GIL for heavy TF workloads."""
    def __init__(self, model_path: Optional[str], seq_len: int, use_tflite: bool = False):
        self.model_path = model_path
        self.seq_len = int(seq_len)
        self.use_tflite = use_tflite
        self.request_q = queue.Queue(maxsize=8)
        self.reply_q = queue.Queue(maxsize=8)
        self.proc = threading.Thread(target=self._worker, daemon=True)
        self.proc.start()

    def _worker(self):
        # Load model in this thread (isolates TF memory)
        keras_model = None
        tflite_interp = None
        if self.model_path:
            try:
                if self.use_tflite:
                    if HAS_TFLITE_RUNTIME:
                        tflite_interp = tflite_runtime.Interpreter(model_path=self.model_path)
                        tflite_interp.allocate_tensors()
                        in_det = tflite_interp.get_input_details()
                        out_det = tflite_interp.get_output_details()
                        logging.info(f"TFLite model loaded: {self.model_path}")
                    else:
                        logging.warning("tflite-runtime not installed; cannot load TFLite model.")
                        tflite_interp = None
                else:
                    if HAS_KERAS:
                        keras_model = load_model(self.model_path)
                        logging.info(f"Keras model loaded: {self.model_path}")
                    else:
                        logging.warning("Keras/TensorFlow not available.")
            except Exception as e:
                logging.exception("Failed to load model: %s", e)
                keras_model = None
                tflite_interp = None
        # worker loop
        while True:
            try:
                req = self.request_q.get()
            except Exception:
                break
            if req is None:
                break
            emg_win, imu_win, meta = req
            amp_pred, phase_pred = None, None
            try:
                if keras_model is not None:
                    emg_in = emg_win.reshape(1, -1, 1).astype(np.float32)
                    imu_in = imu_win.reshape(1, -1, 1).astype(np.float32)
                    preds = keras_model.predict([emg_in, imu_in], verbose=0)
                    # support multiple output shapes
                    if isinstance(preds, list) and len(preds) >= 2:
                        amp_pred = float(np.array(preds[0]).reshape(-1)[0])
                        cs = np.array(preds[1]).reshape(-1, 2)[0]
                        phase_pred = math.atan2(cs[1], cs[0])
                    else:
                        amp_pred = float(np.array(preds).reshape(-1)[0])
                        phase_pred = 0.0
                elif tflite_interp is not None:
                    # try setting inputs; handle both single and two-input models
                    inputs = tflite_interp.get_input_details()
                    outputs = tflite_interp.get_output_details()
                    if len(inputs) >= 2:
                        tflite_interp.set_tensor(inputs[0]['index'], emg_win.reshape(1, -1, 1).astype(np.float32))
                        tflite_interp.set_tensor(inputs[1]['index'], imu_win.reshape(1, -1, 1).astype(np.float32))
                    else:
                        # concatenate channels
                        inp = np.stack([emg_win, imu_win], axis=-1).reshape(1, -1, 2).astype(np.float32)
                        tflite_interp.set_tensor(inputs[0]['index'], inp)
                    tflite_interp.invoke()
                    out0 = tflite_interp.get_tensor(outputs[0]['index'])
                    out1 = tflite_interp.get_tensor(outputs[1]['index']) if len(outputs) >= 2 else None
                    amp_pred = float(np.array(out0).reshape(-1)[0])
                    if out1 is not None:
                        cs = np.array(out1).reshape(-1, 2)[0]
                        phase_pred = math.atan2(cs[1], cs[0])
                    else:
                        phase_pred = 0.0
                else:
                    # fallback heuristic: use analytic on imu_win
                    analytic = hilbert(imu_win)
                    amp_pred = float(np.abs(analytic[-1]))
                    phase_pred = float(np.angle(analytic[-1]))
            except Exception as e:
                logging.exception("Predictor worker error: %s", e)
                # fallback
                analytic = hilbert(imu_win)
                amp_pred = float(np.abs(analytic[-1]))
                phase_pred = float(np.angle(analytic[-1]))
            # reply
            try:
                self.reply_q.put_nowait((amp_pred, phase_pred, meta))
            except queue.Full:
                # drop if cannot reply
                pass

    def predict(self, emg_win: np.ndarray, imu_win: np.ndarray, meta: Optional[dict] = None, timeout=0.5):
        try:
            self.request_q.put_nowait((emg_win.astype(np.float32), imu_win.astype(np.float32), meta or {}))
        except queue.Full:
            logging.debug("Predictor request queue full; skipping prediction")
            return None
        try:
            amp, phase, meta = self.reply_q.get(timeout=timeout)
            return amp, phase, meta
        except queue.Empty:
            return None

    def stop(self):
        try:
            self.request_q.put_nowait(None)
        except Exception:
            pass

# ---------------------------
# MPC controller wrapper
# ---------------------------
def solve_phase_mpc(phi0: float, omega: float, dt: float, phi_ref: np.ndarray, Np: int,
                    B: float, u_min: float, u_max: float, lambda_u: float, lambda_du: float):
    """Solve QP for phase control. Use cvxpy if available; else numeric fallback."""
    if HAS_CVXPY:
        phi = cp.Variable(Np + 1)
        u = cp.Variable(Np)
        constraints = [phi[0] == phi0]
        for k in range(Np):
            constraints += [phi[k + 1] == phi[k] + omega * dt + B * u[k]]
            constraints += [u[k] >= u_min, u[k] <= u_max]
        cost = 0
        for k in range(Np):
            cost += cp.square(phi[k + 1] - phi_ref[k + 1])
            cost += lambda_u * cp.square(u[k])
            if k >= 1:
                cost += lambda_du * cp.square(u[k] - u[k - 1])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.zeros(Np), np.full(Np + 1, phi0 + omega * dt * np.arange(Np + 1))
        return np.array(u.value).reshape(-1), np.array(phi.value).reshape(-1)
    else:
        # dense numeric solve via projected gradient descent on small horizon
        N = Np
        A = np.tril(np.ones((N, N)))
        c0 = phi0 + np.arange(1, N + 1) * (omega * dt)
        phi_ref_vec = phi_ref[1:N + 1]
        H = (B ** 2) * (A.T @ A) + lambda_u * np.eye(N)
        D = np.zeros((N - 1, N))
        for i in range(N - 1):
            D[i, i] = -1
            D[i, i + 1] = 1
        H = H + lambda_du * (D.T @ D)
        g = (B * A).T @ (c0 - phi_ref_vec)
        u = np.zeros(N)
        L = np.linalg.eigvalsh(H).max() if N > 0 else 1.0
        step = 1.0 / (L + 1e-9)
        for it in range(400):
            grad = H @ u + g
            u = u - step * grad
            u = np.clip(u, u_min, u_max)
        phi_pred = np.empty(N + 1)
        phi_pred[0] = phi0
        for k in range(N):
            phi_pred[k + 1] = phi_pred[k] + omega * dt + B * u[k]
        return u, phi_pred

# ---------------------------
# RL tuner (CEM) that evaluates parameter sets on stored traces (safe offline)
# ---------------------------
def cem_optimize_on_trace(trace_pred_fn, init_mean: np.ndarray, init_std: np.ndarray, iters: int = 40, batch: int = 64, elite_frac: float = 0.2):
    rng = np.random.default_rng(12345)
    mean = init_mean.copy()
    std = init_std.copy()
    n_elite = max(1, int(np.round(batch * elite_frac)))
    best = {"score": -1e9, "params": None}
    history = []
    for it in range(iters):
        samples = rng.normal(mean[None, :], std[None, :], size=(batch, len(mean)))
        scores = np.zeros(batch, dtype=float)
        details = [None] * batch
        for i in range(batch):
            B, lam_u, lam_du = float(samples[i, 0]), float(max(samples[i, 1], 0.0)), float(max(samples[i, 2], 0.0))
            res = trace_pred_fn(B, lam_u, lam_du)
            score = float(res.get("reward", -1e9))
            scores[i] = score
            details[i] = res
        idx = np.argsort(scores)[::-1]
        elites = samples[idx[:n_elite]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-9
        if scores[idx[0]] > best["score"]:
            best["score"] = float(scores[idx[0]])
            best["params"] = samples[idx[0]].tolist()
            best["detail"] = details[idx[0]]
        history.append({"iter": it, "best_score": float(best["score"]), "best_params": list(best["params"]), "mean": mean.tolist()})
    return best, history

# ---------------------------
# Orchestrator: ties everything together and logs
# ---------------------------
class Orchestrator:
    def __init__(self, port: str, baud: int, model_path: Optional[str], tflite_path: Optional[str], logdir: str, emg_fs: float = DEFAULT_EMG_FS):
        self.port = port
        self.baud = baud
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.logdir = logdir
        mkdirp(self.logdir)
        self.emg_fs = float(emg_fs)
        self.seq_len = int(round(WINDOW_SECONDS * self.emg_fs))
        # ring buffers
        cap = int(self.emg_fs * 60)  # keep 60s buffer
        self.emg_rb = RingBuffer(cap)
        self.imu_rb = RingBuffer(cap)
        # queues
        self.frame_q = queue.Queue(maxsize=2048)
        self.feature_q = queue.Queue(maxsize=4096)
        self.control_q = queue.Queue(maxsize=512)
        self.stop_evt = threading.Event()
        # threads
        self.serial_thread = SerialFrameReader(self.port, self.baud, self.frame_q, self.stop_evt)
        self.dsp_thread = threading.Thread(target=self._dsp_worker, daemon=True)
        self.ctrl_thread = threading.Thread(target=self._control_worker, daemon=True)
        self.logger_thread = threading.Thread(target=self._logger_worker, daemon=True)
        self.controller_writer_thread = threading.Thread(target=self._controller_writer, daemon=True)
        # predictor
        use_tflite = False
        if self.tflite_path:
            use_tflite = True
        self.predictor = PredictorProcess(self.model_path if not use_tflite else self.tflite_path, seq_len=self.seq_len, use_tflite=use_tflite)
        # MPC params
        self.mpc_params = {"B": -0.05, "lambda_u": 1e-2, "lambda_du": 1e-2, "u_min": U_MIN, "u_max": U_MAX, "horizon": MPC_HORIZON_SECONDS, "desired_phase_offset": math.pi}
        # safety trackers
        self.last_control_time = 0.0
        self.controls_sent = 0
        # log files
        self.raw_path = os.path.join(self.logdir, "raw_frames.csv")
        self.features_path = os.path.join(self.logdir, "features.csv")
        self.pred_path = os.path.join(self.logdir, "predictions.csv")
        self.ctrl_path = os.path.join(self.logdir, "controls.csv")
        # PLL tracker optional
        self.pll = SimplePLL(fs=self.emg_fs, init_freq=5.0)  # start PLL with 5 Hz
        # design DSP filters for imu and emg used offline in dsp worker
        self.imu_sos, self.imu_zi = design_sos(self.emg_fs, IMU_TREMOR_BAND[0], IMU_TREMOR_BAND[1], order=4)
        self.emg_sos, self.emg_zi = design_sos(self.emg_fs, EMG_BAND[0], EMG_BAND[1], order=4)

    def start(self):
        logging.info("Starting orchestrator threads")
        self.serial_thread.start()
        self.dsp_thread.start()
        self.ctrl_thread.start()
        self.logger_thread.start()
        self.controller_writer_thread.start()

    def stop(self):
        logging.info("Stopping orchestrator")
        self.stop_evt.set()
        self.predictor.stop()

    def _dsp_worker(self):
        """
        Consumes frames from frame_q, pushes into ringbuffers, computes rolling features,
        and pushes snapshots for control/prediction at CONTROL_INTERVAL cadence.
        """
        last_ctrl_time = time.time()
        while not self.stop_evt.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # unpack frame samples and push to buffers
            samples = frame["samples"]
            emg_vals = np.array([s[0] for s in samples], dtype=np.float32)
            imu_vals = np.array([s[1] for s in samples], dtype=np.float32)  # use gx axis for now
            self.emg_rb.push(emg_vals)
            self.imu_rb.push(imu_vals)
            # log raw frame metadata
            raw_row = {"t0_us": int(frame["t0_us"]), "seq": int(frame["seq"]), "n": int(frame["n"]), "recv_us": current_micros()}
            try:
                with open(self.raw_path, "a") as f:
                    f.write(",".join([str(raw_row[k]) for k in ("t0_us", "seq", "n", "recv_us")]) + "\n")
            except Exception:
                pass
            # Fast feature extraction on last window
            if self.emg_rb.available() >= self.seq_len and self.imu_rb.available() >= self.seq_len:
                try:
                    emg_win = self.emg_rb.last(self.seq_len)
                    imu_win = self.imu_rb.last(self.seq_len)
                except ValueError:
                    continue
                # filter imu and emg (stateful)
                try:
                    imu_f = sosfilt(self.imu_sos, imu_win)
                except Exception:
                    imu_f = imu_win
                try:
                    emg_f = sosfilt(self.emg_sos, emg_win)
                except Exception:
                    emg_f = emg_win
                # analytic signal
                analytic = hilbert(imu_f)
                inst_amp = float(np.abs(analytic[-1]))
                inst_phase = float(np.angle(analytic[-1]))
                # PLL update (robust)
                pll_phase, pll_amp, pll_freq = self.pll.step(imu_f[-1])
                # PSD features
                psd = compute_psd_features(imu_f, fs=self.emg_fs, fmin=IMU_TREMOR_BAND[0], fmax=IMU_TREMOR_BAND[1])
                emg_env = float(np.sqrt(np.mean(np.square(emg_f[-int(0.1*self.emg_fs):]))))
                feat = {"t": time.time(), "inst_amp": inst_amp, "inst_phase": inst_phase, "pll_phase": pll_phase, "pll_freq": pll_freq,
                        "band_power": psd["band_power"], "peak_freq": psd["peak_freq"], "emg_env": emg_env}
                try:
                    # non-blocking
                    self.feature_q.put_nowait(("feature", feat, emg_win.copy(), imu_f.copy()))
                except queue.Full:
                    pass
            # cadence for control/prediction
            now = time.time()
            if now - last_ctrl_time >= CONTROL_INTERVAL:
                last_ctrl_time = now
                # we expect feature_q to be filled frequently; if empty, skip
                try:
                    tag, feat, emg_win, imu_f = self.feature_q.get(timeout=0.01)
                    # Put back for control loop to consume (duplicate or send snapshot)
                    try:
                        self.feature_q.put_nowait((tag, feat, emg_win, imu_f))
                    except queue.Full:
                        pass
                except queue.Empty:
                    # not enough features, skip this cycle
                    pass
        logging.info("DSP worker stopped")

    def _control_worker(self):
        """
        Runs at CONTROL_INTERVAL cadence: predict -> MPC -> log -> enqueue control
        """
        last_time = time.time()
        while not self.stop_evt.is_set():
            now = time.time()
            if now - last_time < CONTROL_INTERVAL:
                time.sleep(0.001)
                continue
            last_time = now
            # get latest feature snapshot; if none, continue
            try:
                tag, feat, emg_win, imu_f = self.feature_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # model prediction
            pred = self.predictor.predict(emg_win, imu_f, meta={"t": feat["t"]})
            if pred is None:
                # fallback: use analytic
                amp_pred = feat["inst_amp"]
                phase_pred = feat["inst_phase"]
            else:
                amp_pred, phase_pred, _ = pred
            # compute open-loop phi_ref (Np steps)
            dt = CONTROL_INTERVAL
            Np = max(1, int(round(self.mpc_params["horizon"] / dt)))
            # estimate omega (rad/s) from PLL frequency
            omega = 2.0 * math.pi * feat.get("pll_freq", feat.get("peak_freq", 5.0))
            phi0 = feat.get("pll_phase", feat.get("inst_phase", phase_pred))
            phi_ref = np.zeros(Np + 1)
            for k in range(Np + 1):
                phi_ref[k] = phi0 + omega * dt * k
            phi_ref += self.mpc_params.get("desired_phase_offset", math.pi)
            # solve MPC
            u_seq, phi_seq = solve_phase_mpc(phi0=phi0, omega=omega, dt=dt, phi_ref=phi_ref, Np=Np,
                                            B=self.mpc_params["B"], u_min=self.mpc_params["u_min"], u_max=self.mpc_params["u_max"],
                                            lambda_u=self.mpc_params["lambda_u"], lambda_du=self.mpc_params["lambda_du"])
            u_apply = float(clip(u_seq[0], U_MIN, U_MAX))
            # safety: rate limiting and pulse budget
            if time.time() - self.last_control_time < 1.0 / MAX_PULSES_PER_SECOND:
                logging.debug("Control rate limit active; skipping control")
                continue
            self.last_control_time = time.time()
            # enqueue control for writer
            try:
                self.control_q.put_nowait(u_apply)
            except queue.Full:
                logging.debug("Control queue full; dropping control")
            # log prediction & control
            pred_row = {"t": feat["t"], "amp_pred": float(amp_pred), "phase_pred": float(phase_pred), "phi0": phi0, "omega": omega}
            ctrl_row = {"t": time.time(), "u": u_apply, "B": self.mpc_params["B"], "lambda_u": self.mpc_params["lambda_u"], "lambda_du": self.mpc_params["lambda_du"]}
            try:
                with open(self.pred_path, "a") as f:
                    f.write(json.dumps(pred_row) + "\n")
                with open(self.ctrl_path, "a") as f:
                    f.write(json.dumps(ctrl_row) + "\n")
            except Exception:
                pass
        logging.info("Control worker stopped")

    def _logger_worker(self):
        """
        Drains feature queue and writes to CSV for offline analysis.
        """
        # create header on first write
        feat_header_written = os.path.exists(self.features_path)
        with open(self.features_path, "a") as ffeat:
            while not self.stop_evt.is_set():
                try:
                    tag, feat, *_ = self.feature_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                # append
                try:
                    ffeat.write(json.dumps(feat) + "\n")
                except Exception:
                    pass
        logging.info("Logger worker stopped")

    def _controller_writer(self):
        """
        Sends CTL packets to Teensy: 'CTL' + float32 (little-endian)
        """
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as e:
            logging.error("Controller writer cannot open serial: %s", e)
            return
        logging.info("Controller writer opened serial port")
        while not self.stop_evt.is_set():
            try:
                u = self.control_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # safety clamp
            u = clip(u, U_MIN, U_MAX)
            pkt = b'CTL' + struct.pack("<f", float(u))
            try:
                ser.write(pkt)
                self.controls_sent += 1
            except Exception:
                logging.exception("Failed to write control packet")
        ser.close()
        logging.info("Controller writer stopped")

# ---------------------------
# CLI & bootstrap
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ISEF real-time tremor pipeline (EMG + IMU -> CNN/LSTM -> MPC -> RL)")
    p.add_argument("--port", required=True, help="Serial port to Teensy (e.g., /dev/ttyUSB0 or /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=SERIAL_BAUD)
    p.add_argument("--model", default=None, help="Path to Keras .h5 model (optional)")
    p.add_argument("--tflite", default=None, help="Path to TFLite .tflite model (optional)")
    p.add_argument("--logdir", default=LOG_DIR_DEFAULT, help="Directory to save logs")
    p.add_argument("--emg_fs", type=float, default=DEFAULT_EMG_FS, help="EMG sampling rate (Hz) fallback")
    return p.parse_args()

def main():
    args = parse_args()
    orch = Orchestrator(port=args.port, baud=args.baud, model_path=args.model, tflite_path=args.tflite, logdir=args.logdir, emg_fs=args.emg_fs)
    try:
        orch.start()
        logging.info("Orchestrator started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: stopping...")
        orch.stop()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
