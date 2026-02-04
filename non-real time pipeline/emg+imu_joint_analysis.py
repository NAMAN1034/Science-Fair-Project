#!/usr/bin/env python3
"""
emg_imu_joint_analysis.py
- Loads emg_raw_capture.csv (time_us, emg_raw) and imu_capture.csv (ax,ay,az,gx,gy,gz,time)
- Aligns IMU amplitude to EMG timestamps (interpolation)
- Computes EMG: bandpass(20-250Hz) -> rect -> RMS window (100 ms) -> slope
- Computes IMU tremor amplitude+phase using gyro axis hilbert (calls similar logic)
- Cross-correlation to estimate lead time (EMG leads IMU if positive)
- Computes Phase-Locking Value (PLV) between EMG bursts and IMU tremor phase
- Exports emg_imu_features.csv: time_s, EMG_RMS, EMG_Slope, IMU_Amplitude, IMU_Phase
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, correlate, stft, welch
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

EMG_CSV = "emg_raw_capture.csv"
IMU_CSV = "imu_capture.csv"

# ---------- utility filters ----------
def bandpass(signal, fs, low, high, order=4):
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,signal)

# ---------- PLV calculation ----------
def phase_locking_value(phases1, phases2):
    # both arrays same length
    complex_phase = np.exp(1j*(phases1 - phases2))
    return np.abs(np.mean(complex_phase))

# ---------- main ----------
def main():
    emg_df = pd.read_csv(EMG_CSV, header=0)
    imu_df = pd.read_csv(IMU_CSV, header=0)
    # EMG: first two columns assumed time_us, emg_raw
    emg_time = emg_df.iloc[:,0].astype(float).values * 1e-6
    emg_raw = emg_df.iloc[:,1].astype(float).values

    # IMU
    imu_time_raw = imu_df["time"].astype(float).values
    # detect units similar to prior script
    dt = np.median(np.diff(imu_time_raw))
    if dt > 1e3:
        imu_time = imu_time_raw * 1e-6
    elif dt > 1:
        imu_time = imu_time_raw * 1e-3
    else:
        imu_time = imu_time_raw 
    gx = imu_df["gx"].astype(float).values
    gy = imu_df["gy"].astype(float).values
    gz = imu_df["gz"].astype(float).values

    # IMU tremor axis selection (use gz by default, or choose best)
    imu_amp_raw = np.sqrt(gx**2 + gy**2 + gz**2)
    # interpolate imu amplitude onto emg timestamps
    interp = interp1d(imu_time, imu_amp_raw, kind='linear', fill_value='extrapolate')
    imu_amp_on_emg = interp(emg_time)

    # Compute IMU phase using bandpass of chosen gyro axis (choose gz)
    fs_imu = 1.0/np.median(np.diff(imu_time))
    # choose gz for phase extraction
    gz_centered = gz - np.mean(gz)
    gz_bp = bandpass(gz_centered, fs_imu, 4.0, 8.0)
    analytic_gz = hilbert(gz_bp)
    imu_phase_time = np.unwrap(np.angle(analytic_gz))
    # interpolate phase to emg_time
    interp_phase = interp1d(imu_time, imu_phase_time, kind='linear', fill_value='extrapolate')
    imu_phase_on_emg = interp_phase(emg_time)

    # EMG processing
    fs_emg = 1.0 / np.median(np.diff(emg_time))
    emg_bp = bandpass(emg_raw, fs_emg, 20, 250)
    emg_rect = np.abs(emg_bp)
    window_samples = max(1, int(fs_emg * 0.1))  # 100 ms
    emg_rms = np.sqrt(uniform_filter1d(emg_rect**2, size=window_samples))
    emg_slope = np.gradient(emg_rms, emg_time)

    # Cross-correlation
    corr = correlate(emg_rms - np.mean(emg_rms), imu_amp_on_emg - np.mean(imu_amp_on_emg), mode='full')
    lags = np.arange(-len(emg_rms)+1, len(emg_rms))
    lag_sec = lags / fs_emg
    max_idx = np.argmax(corr)
    lead_time = lag_sec[max_idx]

    # Compute PLV between EMG burst times and IMU phase: approach -> detect EMG peaks (threshold) and sample phase at those times
    # Detect EMG peaks by threshold on RMS (mean + k*std)
    thresh = np.mean(emg_rms) + 1.0 * np.std(emg_rms)
    peak_idx = np.where(emg_rms > thresh)[0]
    if len(peak_idx) > 0:
        phases_at_peaks = imu_phase_on_emg[peak_idx]
        # For EMG we don't have instantaneous phase, so use the phase array mod 2pi
        # PLV: mean vector length of exp(j*phase)
        plv = np.abs(np.mean(np.exp(1j*phases_at_peaks)))
    else:
        plv = np.nan

    print(f"Lead time (EMG->IMU): {lead_time:.3f} s")
    print(f"PLV (EMG bursts vs IMU phase): {plv}")

    # export features
    out = pd.DataFrame({
        "time_s": emg_time,
        "EMG_RMS": emg_rms,
        "EMG_Slope": emg_slope,
        "IMU_Amplitude": imu_amp_on_emg,
        "IMU_Phase": imu_phase_on_emg
    })
    out.to_csv("emg_imu_features.csv", index=False)
    print("Saved emg_imu_features.csv")

    # Plots
    plt.figure(figsize=(14,10))
    plt.subplot(4,1,1)
    plt.plot(emg_time, emg_raw); plt.title("Raw EMG")
    plt.subplot(4,1,2)
    plt.plot(emg_time, emg_rms); plt.title("EMG RMS")
    plt.subplot(4,1,3)
    plt.plot(emg_time, imu_amp_on_emg); plt.title("IMU amplitude (aligned)")
    plt.subplot(4,1,4)
    plt.plot(emg_time, imu_phase_on_emg); plt.title("IMU phase (aligned)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
