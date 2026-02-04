import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d


# Load CSV (robust to missing headers)
df = pd.read_csv("emg_raw_capture.csv", header=None)

print("Detected CSV columns:")
print(df.head())

# Expecting:
# column 0 → timestamp (microseconds)
# column 1 → EMG raw value

if df.shape[1] < 2:
    raise ValueError("CSV must have at least 2 columns: time_us, emg_raw")

time_us = df.iloc[:, 0].values.astype(float)
emg_raw = df.iloc[:, 1].values.astype(float)

# Time & Sampling Rate
time_s = (time_us - time_us[0]) * 1e-6

fs = 1.0 / np.mean(np.diff(time_s))
print(f"Estimated sampling rate: {fs:.2f} Hz")

# Band-pass Filter (20–250 Hz)
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

emg_bandpassed = bandpass_filter(emg_raw, 20, 250, fs)

# Rectification
emg_rectified = np.abs(emg_bandpassed)

# RMS Envelope (100 ms)
window_ms = 100
window_samples = int(fs * window_ms / 1000)

emg_rms = np.sqrt(
    uniform_filter1d(emg_rectified ** 2, size=window_samples)
)

# Plot Results
plt.figure(figsize=(14, 9))

plt.subplot(3, 1, 1)
plt.plot(time_s, emg_raw, linewidth=0.6)
plt.title("Raw EMG Signal")
plt.ylabel("ADC Counts")
plt.grid(False)

plt.subplot(3, 1, 2)
plt.plot(time_s, emg_bandpassed, linewidth=0.6)
plt.title("Band-Passed EMG (20–250 Hz)")
plt.ylabel("Counts")
plt.grid(False)

plt.subplot(3, 1, 3)
plt.plot(time_s, emg_rms, linewidth=2)
plt.title("EMG RMS Envelope (Muscle Activation)")
plt.xlabel("Time (s)")
plt.ylabel("RMS Amplitude")
plt.grid(False)

plt.tight_layout()
plt.show()

from scipy.signal import welch, stft

# Power Spectral Density (Welch)
freqs, psd = welch(
    emg_bandpassed,
    fs=fs,
    nperseg=int(fs * 2)  # 2-second windows
)

# Tremor band power (4–7 Hz)
tremor_band = (freqs >= 4) & (freqs <= 7)
tremor_power = np.sum(psd[tremor_band])

print(f"Tremor-band EMG power (4–7 Hz): {tremor_power:.2e}")

# Spectrogram (STFT)
f, t, Zxx = stft(
    emg_bandpassed,
    fs=fs,
    nperseg=int(fs * 0.5),
    noverlap=int(fs * 0.4)
)

# Plot PSD
plt.figure(figsize=(10, 4))
plt.semilogy(freqs, psd)
plt.axvspan(4, 7, color="red", alpha=0.2, label="Tremor Band")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("EMG Power Spectral Density")
plt.legend()
plt.grid(True)
plt.show()

# Plot Spectrogram
plt.figure(figsize=(12, 5))
plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
plt.ylim(0, 20)
plt.colorbar(label="Amplitude")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("EMG Time–Frequency Representation")
plt.show()
