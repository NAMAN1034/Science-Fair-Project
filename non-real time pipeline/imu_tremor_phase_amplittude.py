#!/usr/bin/env python3
"""
imu_tremor_phase_amplitude.py
- Loads IMU CSV (expects columns: ax,ay,az,gx,gy,gz,time)
- Auto-detects sampling rate and time units
- Selects best gyro axis (highest PSD in tremor band)
- Band-pass filters chosen axis into tremor band (default 4-8 Hz)
- Computes Hilbert amplitude and phase (instantaneous)
- Saves imu_tremor_features.csv with time_s, axis, tremor_signal, tremor_amplitude, tremor_phase
- Plots raw, filtered, amplitude, phase, PSD
Usage: python imu_tremor_phase_amplitude.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, hilbert

CSV = "imu_capture.csv"   # adjust if needed
TREMOR_LOW = 4.0
TREMOR_HIGH = 8.0
AXES = ["gx", "gy", "gz"]  # prefer gyros for tremor

def load_imu(csv_path):
    df = pd.read_csv(csv_path, header=0)
    required = ["ax","ay","az","gx","gy","gz","time"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column {r} in {csv_path}")
    time_raw = df["time"].values.astype(float)
    # infer units
    dt = np.median(np.diff(time_raw))
    if dt > 1e3:
        time_s = time_raw * 1e-6
    elif dt > 1:
        time_s = time_raw * 1e-3
    else:
        time_s = time_raw
    df["time_s"] = time_s
    return df

def bandpass(data, fs, low, high, order=4):
    nyq = 0.5*fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,data)

def choose_best_axis(df, fs, low, high):
    # compute PSD for each axis, pick highest integrated power in band
    best = None
    best_power = -1
    freqs = None
    for ax in AXES:
        f, pxx = welch(df[ax].values.astype(float), fs=fs, nperseg=min(2048, len(df)))
        if freqs is None:
            freqs = f
        band = (f >= low) & (f <= high)
        power = np.sum(pxx[band])
        if power > best_power:
            best_power = power
            best = ax
    return best, freqs

def main():
    df = load_imu(CSV)
    time_s = df["time_s"].values
    fs = 1.0 / np.median(np.diff(time_s))
    print(f"Detected sampling rate: {fs:.1f} Hz")
    best_axis, freqs = choose_best_axis(df, fs, TREMOR_LOW, TREMOR_HIGH)
    print("Best gyro axis (tremor power):", best_axis)

    raw_signal = df[best_axis].values.astype(float)
    raw_signal = raw_signal - np.mean(raw_signal)

    # bandpass to tremor band
    tremor_sig = bandpass(raw_signal, fs, TREMOR_LOW, TREMOR_HIGH)

    # analytic signal
    analytic = hilbert(tremor_sig)
    tremor_amp = np.abs(analytic)
    tremor_phase = np.unwrap(np.angle(analytic))

    # PSD and dominant frequency
    f, pxx = welch(tremor_sig, fs=fs, nperseg=min(2048, len(tremor_sig)))
    dom_idx = np.argmax(pxx[(f>=TREMOR_LOW)&(f<=TREMOR_HIGH)])
    dom_freq = f[(f>=TREMOR_LOW)&(f<=TREMOR_HIGH)][dom_idx]

    # Save features
    out = pd.DataFrame({
        "time_s": time_s,
        "axis": best_axis,
        "tremor_signal": tremor_sig,
        "tremor_amplitude": tremor_amp,
        "tremor_phase": tremor_phase
    })
    out.to_csv("imu_tremor_features.csv", index=False)
    print("Saved imu_tremor_features.csv, dominant tremor freq (Hz):", dom_freq)

    # plots
    plt.figure(figsize=(12,10))
    plt.subplot(4,1,1)
    plt.plot(time_s, raw_signal); plt.title("Raw selected gyro axis"); plt.grid(True)
    plt.subplot(4,1,2)
    plt.plot(time_s, tremor_sig); plt.title(f"Bandpassed {TREMOR_LOW}-{TREMOR_HIGH} Hz"); plt.grid(True)
    plt.subplot(4,1,3)
    plt.plot(time_s, tremor_amp); plt.title("Tremor amplitude (analytic)"); plt.grid(True)
    plt.subplot(4,1,4)
    plt.plot(time_s, tremor_phase); plt.title("Tremor phase (unwrapped)"); plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
