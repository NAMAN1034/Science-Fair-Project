import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. LOAD AND CLEAN DATA
FILENAME = "imu_capture.csv"
try:
    data = pd.read_csv(FILENAME)
    data.columns = ["ax", "ay", "az", "gx", "gy", "gz", "time"]
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# SORTING: Prevents back-and-forth lines in plots caused by jittery timestamps
data = data.sort_values("time").reset_index(drop=True)

# 2. AUTO-CALCULATE SAMPLING RATE (fs)
time_diffs = data["time"].diff().dropna()
avg_dt = time_diffs.mean()
fs = 1.0 / avg_dt
print(f"Detected Sampling Rate: {fs:.2f} Hz")

# Relative time starting at 0
t = (data["time"] - data["time"].iloc[0]).values

# 3. BANDPASS FILTER (Tremor Range: 3-12 Hz)
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

ax_filt = bandpass_filter(data["ax"], 3, 12, fs)
gz_filt = bandpass_filter(data["gz"], 3, 12, fs)

# 4. SLIDING WINDOW RMS (AMPLITUDE OVER TIME)
window_size = int(fs * 0.5) 
def sliding_rms(signal, win):
    return np.sqrt(np.convolve(signal**2, np.ones(win)/win, mode='valid'))

ax_rms = sliding_rms(ax_filt, window_size)
gz_rms = sliding_rms(gz_filt, window_size)
rms_time = t[window_size - 1:] # Align time with convolution offset

# GRAPH 1: RMS Amplitude
plt.figure(figsize=(10, 5))
plt.plot(rms_time, ax_rms, label="Accel X RMS")
plt.plot(rms_time, gz_rms, label="Gyro Z RMS")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Sliding Window RMS (Tremor Intensity)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. FFT ANALYSIS (FREQUENCY CONTENT)
def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal)) / n
    return freqs, fft_vals

freqs, fft_ax = compute_fft(ax_filt, fs)
_, fft_gz = compute_fft(gz_filt, fs)

# GRAPH 2: FFT Frequency Spectrum
plt.figure(figsize=(10, 5))
plt.plot(freqs, fft_ax, label="Accel X FFT")
plt.plot(freqs, fft_gz, label="Gyro Z FFT")
plt.xlim(0, 20)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("FFT of Filtered IMU Signals (Frequency Peaks)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. PHASE DIFFERENCE
# We use the full FFT to preserve phase information
fft_ax_full = np.fft.fft(ax_filt)
fft_gz_full = np.fft.fft(gz_filt)
phase_diff = np.angle(fft_ax_full) - np.angle(fft_gz_full)

# GRAPH 3: Phase Difference
plt.figure(figsize=(10, 4))
plt.plot(freqs, phase_diff[:len(freqs)])
plt.xlim(0, 20)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase Difference [rad]")
plt.title("Phase Difference between Accel X and Gyro Z")
plt.grid(True, alpha=0.3)
plt.show()

# 7. PRINT PEAK FREQUENCY
tremor_band = (freqs >= 4) & (freqs <= 12)
if any(tremor_band):
    peak_accel_freq = freqs[tremor_band][np.argmax(fft_ax[tremor_band])]
    print(f"Dominant Tremor Frequency: {peak_accel_freq:.2f} Hz")