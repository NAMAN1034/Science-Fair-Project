"""
lstm_state_estimator_fixed.py

Multimodal CNN+LSTM tremor state estimator (complete, robust, ISEF-ready).

Usage:
  python lstm_state_estimator_fixed.py --train
  python lstm_state_estimator_fixed.py --predict

Inputs expected (flexible):
  - emg_raw_capture.csv  (columns: time_us or time, emg_raw or second column)
  - imu_capture.csv      (columns: ax,ay,az,gx,gy,gz,time OR similar)

Outputs:
  - Trained model saved at --model_out (default: lstm_state_estimator_fixed.h5)
  - Normalization stats saved at <model_out>_norm.npy
  - Predictions saved as <model_out>_predictions.csv when using --predict

Dependencies:
  numpy, pandas, scipy, matplotlib (optional), tensorflow
"""

from __future__ import annotations
import os
import argparse
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.interpolate import interp1d

# Try importing TensorFlow / Keras
try:
    import tensorflow as tf
    from keras import layers, models, optimizers, callbacks
except Exception as e:
    raise ImportError("TensorFlow (tf) is required. Install with `pip install tensorflow`.\n" + str(e))


# -----------------------------
# Configurable defaults
# -----------------------------
EMG_FS_OVERRIDE = 1000.0       # Hz to use if EMG timestamps are invalid (typical)
EMG_BP = (20.0, 250.0)         # EMG bandpass
IMU_TREMOR_BAND = (4.0, 8.0)   # Tremor band (Hz)
DEFAULT_SEQ_SECONDS = 0.5
DEFAULT_PRED_HORIZON = 0.1
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 40


# -----------------------------
# Utilities
# -----------------------------
def infer_time_seconds(arr: np.ndarray) -> np.ndarray:
    """Infer whether arr is microseconds/milliseconds/seconds, return seconds."""
    arr = np.asarray(arr).astype(float)
    if arr.size < 2:
        raise ValueError("Time array too short to infer units.")
    dt = np.median(np.diff(arr))
    if dt > 1e3:
        return arr * 1e-6
    elif dt > 1:
        return arr * 1e-3
    else:
        return arr


def infer_fs(time_s: np.ndarray) -> float | None:
    """Return sampling frequency from time (seconds). None if invalid."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 10:
        return None
    return 1.0 / np.median(dt)


def bandpass_filt(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter using butterworth."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Invalid bandpass edges for fs={fs:.1f} Hz: low_n={low_n:.6f}, high_n={high_n:.6f}")
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x, axis=0)


# -----------------------------
# IO & Preprocessing
# -----------------------------
def load_csv_flex(path: str) -> pd.DataFrame:
    """Load CSV and lowercase/trims column names for flexible parsing."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_and_preprocess(emg_csv: str = "emg_raw_capture.csv",
                        imu_csv: str = "imu_capture.csv",
                        emg_bp: Tuple[float, float] = EMG_BP,
                        imu_tremor_band: Tuple[float, float] = IMU_TREMOR_BAND,
                        emg_fs_override: float = EMG_FS_OVERRIDE) -> Dict:
    """
    Loads EMG and IMU CSVs, normalizes, filters, chooses best gyro axis,
    interpolates IMU axis/amp/phase to EMG timebase, returns dict of arrays.
    """
    # --- EMG load ---
    emg_df = load_csv_flex(emg_csv)
    # Try common column names or fall back to first two columns
    if "time_us" in emg_df.columns:
        emg_time_raw = emg_df["time_us"].values
    elif "time" in emg_df.columns:
        emg_time_raw = emg_df["time"].values
    else:
        emg_time_raw = emg_df.iloc[:, 0].values

    if "emg_raw" in emg_df.columns:
        emg_raw = emg_df["emg_raw"].values.astype(float)
    else:
        emg_raw = emg_df.iloc[:, 1].values.astype(float)

    # Infer time units
    try:
        emg_time_s = infer_time_seconds(emg_time_raw)
        fs_emg = infer_fs(emg_time_s)
        if fs_emg is None or fs_emg < 100:
            raise ValueError("EMG fs invalid")
        print(f"[IO] EMG sampling rate inferred: {fs_emg:.1f} Hz")
    except Exception:
        fs_emg = float(emg_fs_override)
        emg_time_s = np.arange(len(emg_raw)) / fs_emg
        print(f"[IO] EMG timestamps invalid — overriding fs to {fs_emg:.1f} Hz and synthesizing timebase")

    # Band-pass EMG (normalize after filtering)
    emg_bp_signal = bandpass_filt(emg_raw.astype(float), fs_emg, emg_bp[0], emg_bp[1])
    emg_bp_norm = (emg_bp_signal - np.mean(emg_bp_signal)) / (np.std(emg_bp_signal) + 1e-12)

    # --- IMU load ---
    imu_df = load_csv_flex(imu_csv)
    # Accept either "time" or "time_us" as IMU timestamp (or last column)
    if "time" in imu_df.columns:
        imu_time_raw = imu_df["time"].values
    elif "time_us" in imu_df.columns:
        imu_time_raw = imu_df["time_us"].values
    else:
        imu_time_raw = imu_df.iloc[:, -1].values

    imu_time_s = infer_time_seconds(imu_time_raw)
    fs_imu = infer_fs(imu_time_s)
    if fs_imu is None:
        raise ValueError("IMU timestamps invalid or too sparse. Need real IMU timestamps.")
    print(f"[IO] IMU sampling rate inferred: {fs_imu:.1f} Hz")

    # Detect gyro columns robustly
    if all(k in imu_df.columns for k in ("gx", "gy", "gz")):
        gx = imu_df["gx"].astype(float).values
        gy = imu_df["gy"].astype(float).values
        gz = imu_df["gz"].astype(float).values
    else:
        # fallback to positional columns: expect ax,ay,az,gx,gy,gz,time
        cols = imu_df.columns.tolist()
        if len(cols) >= 7:
            gx = imu_df.iloc[:, 3].astype(float).values
            gy = imu_df.iloc[:, 4].astype(float).values
            gz = imu_df.iloc[:, 5].astype(float).values
        else:
            raise ValueError("IMU CSV lacks gyro columns and cannot infer them.")

    # Choose best gyro axis by tremor-band power
    def band_power(series: np.ndarray):
        f, pxx = welch(series.astype(float), fs=fs_imu, nperseg=min(2048, len(series)))
        mask = (f >= imu_tremor_band[0]) & (f <= imu_tremor_band[1])
        return np.sum(pxx[mask])

    p_gx = band_power(gx)
    p_gy = band_power(gy)
    p_gz = band_power(gz)
    axis_powers = {"gx": p_gx, "gy": p_gy, "gz": p_gz}
    best_axis = max(axis_powers, key=axis_powers.get)
    print(f"[IO] best gyro axis by tremor-band power: {best_axis}")

    axis_map = {"gx": gx, "gy": gy, "gz": gz}
    chosen_gyro = axis_map[best_axis].astype(float)
    chosen_centered = chosen_gyro - np.mean(chosen_gyro)
    chosen_tremor = bandpass_filt(chosen_centered, fs_imu, imu_tremor_band[0], imu_tremor_band[1])

    # Analytic signal to get amplitude & phase (on IMU timeline)
    analytic = hilbert(chosen_tremor)
    imu_amp = np.abs(analytic)
    imu_phase = np.angle(analytic)  # wrapped phase

    # Interpolate IMU-derived quantities onto EMG timebase
    interp_axis = interp1d(imu_time_s, chosen_tremor, kind="linear", fill_value="extrapolate")
    interp_amp = interp1d(imu_time_s, imu_amp, kind="linear", fill_value="extrapolate")
    interp_phase = interp1d(imu_time_s, imu_phase, kind="linear", fill_value="extrapolate")

    imu_axis_on_emg = interp_axis(emg_time_s)
    imu_amp_on_emg = interp_amp(emg_time_s)
    imu_phase_on_emg = interp_phase(emg_time_s)

    # Return everything
    return {
        "emg_time_s": emg_time_s,
        "emg_signal": emg_bp_norm,
        "fs_emg": fs_emg,
        "imu_time_s": imu_time_s,
        "fs_imu": fs_imu,
        "imu_axis_on_emg": imu_axis_on_emg,
        "imu_amp_on_emg": imu_amp_on_emg,
        "imu_phase_on_emg": imu_phase_on_emg,
        "best_gyro_axis": best_axis
    }


# -----------------------------
# Windowing - index-based deterministic windows
# -----------------------------
def make_windows_indexed(data: Dict,
                         seq_seconds: float = DEFAULT_SEQ_SECONDS,
                         pred_horizon_seconds: float = DEFAULT_PRED_HORIZON,
                         stride_seconds: float | None = None):
    """Create stacked windows with identical shapes for model input and targets."""
    emg = data["emg_signal"]
    imu_axis = data["imu_axis_on_emg"]
    imu_amp = data["imu_amp_on_emg"]
    imu_phase = data["imu_phase_on_emg"]
    t = data["emg_time_s"]
    fs = data["fs_emg"]

    seq_len = int(round(seq_seconds * fs))
    horizon = int(round(pred_horizon_seconds * fs))
    if stride_seconds is None:
        stride = max(1, seq_len // 4)
    else:
        stride = int(round(stride_seconds * fs))

    N = len(emg)
    max_start = N - seq_len - horizon
    if max_start <= 0:
        raise ValueError("Recording too short for requested seq_seconds and pred_horizon.")

    emg_wins = []
    imu_wins = []
    targets_amp = []
    targets_phase = []
    centers = []

    for start in range(0, max_start + 1, stride):
        end = start + seq_len
        target_idx = end + horizon - 1
        if target_idx >= N:
            continue
        emg_win = emg[start:end]
        imu_win = imu_axis[start:end]
        if emg_win.shape[0] != seq_len or imu_win.shape[0] != seq_len:
            continue
        emg_wins.append(emg_win)
        imu_wins.append(imu_win)
        targets_amp.append(imu_amp[target_idx])
        targets_phase.append(imu_phase[target_idx])
        centers.append(t[start + seq_len // 2])

    emg_wins = np.stack(emg_wins).astype(np.float32)
    imu_wins = np.stack(imu_wins).astype(np.float32)
    targets_amp = np.array(targets_amp).astype(np.float32)
    targets_phase = np.array(targets_phase).astype(np.float32)
    targets_phase_cs = np.stack([np.cos(targets_phase), np.sin(targets_phase)], axis=1).astype(np.float32)

    print(f"[WINDOWS] Created {emg_wins.shape[0]} windows | seq_len={seq_len} | horizon={horizon} | stride={stride}")
    return emg_wins, imu_wins, targets_amp, targets_phase_cs, np.array(centers)


# -----------------------------
# Model definition (Keras)
# -----------------------------
def build_cnn_lstm(seq_len: int, emg_channels: int = 1, imu_channels: int = 1,
                   cnn_filters: int = 32, lstm_units: int = 64) -> tf.keras.Model:
    """
    Two-branch CNN (per-sequence) -> LSTM fusion model.
    Inputs: emg_input (seq_len,1), imu_input (seq_len,1)
    Outputs: amp_out (scalar), phase_cs_out (2)
    """
    emg_input = layers.Input(shape=(seq_len, emg_channels), name="emg_input")
    x = layers.Conv1D(cnn_filters, 7, padding="same", activation="relu")(emg_input)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(cnn_filters*2, 5, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dense(64, activation="relu")(x)

    imu_input = layers.Input(shape=(seq_len, imu_channels), name="imu_input")
    y = layers.Conv1D(cnn_filters, 7, padding="same", activation="relu")(imu_input)
    y = layers.MaxPool1D(2)(y)
    y = layers.Conv1D(cnn_filters*2, 5, padding="same", activation="relu")(y)
    y = layers.MaxPool1D(2)(y)
    y = layers.LSTM(lstm_units, return_sequences=False)(y)
    y = layers.Dense(64, activation="relu")(y)

    fused = layers.Concatenate()([x, y])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.2)(fused)
    fused = layers.Dense(64, activation="relu")(fused)

    amp_out = layers.Dense(1, name="amp_out")(fused)
    phase_cs_out = layers.Dense(2, name="phase_cs_out")(fused)

    model = models.Model(inputs=[emg_input, imu_input], outputs=[amp_out, phase_cs_out], name="cnn_lstm_state_estimator")
    return model


# -----------------------------
# Training, saving, predicting
# -----------------------------
def make_tf_dataset(emg_windows, imu_windows, amp_targets, phase_cs_targets, batch_size=DEFAULT_BATCH, shuffle=True):
    emg_in = emg_windows[..., np.newaxis]
    imu_in = imu_windows[..., np.newaxis]
    ds = tf.data.Dataset.from_tensor_slices(((emg_in, imu_in), (amp_targets, phase_cs_targets)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(16384, len(emg_windows)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compile_and_train(model: tf.keras.Model, train_ds, val_ds, model_out: str, epochs: int = DEFAULT_EPOCHS):
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss={"amp_out": "mse", "phase_cs_out": "mse"},
                  loss_weights={"amp_out": 1.0, "phase_cs_out": 0.5})
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_out, monitor="val_loss", save_best_only=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb, verbose=2)
    return history


# -----------------------------
# High-level train and predict flows
# -----------------------------
def train_flow(args):
    data = load_and_preprocess(emg_csv=args.emg_csv, imu_csv=args.imu_csv,
                               emg_bp=EMG_BP, imu_tremor_band=IMU_TREMOR_BAND,
                               emg_fs_override=args.emg_fs_override)

    emg_wins, imu_wins, amp_t, phase_t_cs, centers = make_windows_indexed(data,
                                                                           seq_seconds=args.seq_seconds,
                                                                           pred_horizon_seconds=args.pred_horizon,
                                                                           stride_seconds=None)
    if len(emg_wins) < 10:
        raise ValueError("Too few windows for training. Record longer or reduce seq_seconds.")

    # train/val split
    N = len(emg_wins)
    split = int(N * 0.8)
    train_emg, val_emg = emg_wins[:split], emg_wins[split:]
    train_imu, val_imu = imu_wins[:split], imu_wins[split:]
    train_amp, val_amp = amp_t[:split], amp_t[split:]
    train_phase, val_phase = phase_t_cs[:split], phase_t_cs[split:]

    # normalization
    emg_mean, emg_std = float(np.mean(train_emg)), float(np.std(train_emg)) + 1e-12
    imu_mean, imu_std = float(np.mean(train_imu)), float(np.std(train_imu)) + 1e-12
    train_emg_n = (train_emg - emg_mean) / emg_std
    val_emg_n = (val_emg - emg_mean) / emg_std
    train_imu_n = (train_imu - imu_mean) / imu_std
    val_imu_n = (val_imu - imu_mean) / imu_std

    amp_mean, amp_std = float(np.mean(train_amp)), float(np.std(train_amp)) + 1e-12
    train_amp_n = (train_amp - amp_mean) / amp_std
    val_amp_n = (val_amp - amp_mean) / amp_std

    # tf datasets
    train_ds = make_tf_dataset(train_emg_n, train_imu_n, train_amp_n, train_phase, batch_size=args.batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_emg_n, val_imu_n, val_amp_n, val_phase, batch_size=args.batch_size, shuffle=False)

    seq_len = train_emg.shape[1]
    model = build_cnn_lstm(seq_len, emg_channels=1, imu_channels=1, cnn_filters=32, lstm_units=64)
    print("[MODEL] Summary:")
    model.summary()

    print("[TRAIN] Starting training...")
    compile_and_train(model, train_ds, val_ds, model_out=args.model_out, epochs=args.epochs)

    # save normalization stats
    norm = {"emg_mean": emg_mean, "emg_std": emg_std,
            "imu_mean": imu_mean, "imu_std": imu_std,
            "amp_mean": amp_mean, "amp_std": amp_std,
            "seq_seconds": args.seq_seconds, "pred_horizon": args.pred_horizon}
    np.save(args.model_out + "_norm.npy", norm)
    print(f"[I/O] Model and normalization saved: {args.model_out}, {args.model_out}_norm.npy")


def predict_flow(args):
    if not os.path.exists(args.model_out) or not os.path.exists(args.model_out + "_norm.npy"):
        raise FileNotFoundError("Model or normalization stats not found. Train first.")

    norm = np.load(args.model_out + "_norm.npy", allow_pickle=True).item()
    data = load_and_preprocess(emg_csv=args.emg_csv, imu_csv=args.imu_csv,
                               emg_bp=EMG_BP, imu_tremor_band=IMU_TREMOR_BAND,
                               emg_fs_override=args.emg_fs_override)

    emg_wins, imu_wins, amp_t, phase_t_cs, centers = make_windows_indexed(data,
                                                                           seq_seconds=norm["seq_seconds"],
                                                                           pred_horizon_seconds=norm["pred_horizon"],
                                                                           stride_seconds=None)
    if len(emg_wins) == 0:
        print("[PREDICT] No windows generated. Check recording length and seq_seconds.")
        return

    emg_wins_n = (emg_wins - norm["emg_mean"]) / norm["emg_std"]
    imu_wins_n = (imu_wins - norm["imu_mean"]) / norm["imu_std"]

    model = tf.keras.models.load_model(args.model_out, compile=False)
    emg_in = emg_wins_n[..., np.newaxis]
    imu_in = imu_wins_n[..., np.newaxis]
    preds = model.predict((emg_in, imu_in), batch_size=args.batch_size, verbose=1)
    amp_pred_n = preds[0].reshape(-1)
    phase_cs_pred = preds[1]
    amp_pred = amp_pred_n * norm["amp_std"] + norm["amp_mean"]
    phase_pred = np.arctan2(phase_cs_pred[:, 1], phase_cs_pred[:, 0])

    out = pd.DataFrame({
        "time_s": centers,
        "amp_pred": amp_pred,
        "phase_pred": phase_pred,
        "amp_target": amp_t,
        "phase_target_cos": phase_t_cs[:, 0],
        "phase_target_sin": phase_t_cs[:, 1]
    })
    out_csv = os.path.splitext(args.model_out)[0] + "_predictions.csv"
    out.to_csv(out_csv, index=False)
    print(f"[PREDICT] Saved predictions to {out_csv}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Train model from CSVs")
    p.add_argument("--predict", action="store_true", help="Run prediction using saved model")
    p.add_argument("--emg_csv", default="emg_raw_capture.csv", help="EMG CSV path")
    p.add_argument("--imu_csv", default="imu_capture.csv", help="IMU CSV path")
    p.add_argument("--model_out", default="lstm_state_estimator_fixed.keras", help="Model output path")    
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--seq_seconds", type=float, default=DEFAULT_SEQ_SECONDS)
    p.add_argument("--pred_horizon", type=float, default=DEFAULT_PRED_HORIZON)
    p.add_argument("--emg_fs_override", type=float, default=EMG_FS_OVERRIDE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train_flow(args)
    elif args.predict:
        predict_flow(args)
    else:
        print("Usage examples:")
        print("  python lstm_state_estimator_fixed.py --train")
        print("  python lstm_state_estimator_fixed.py --predict")
