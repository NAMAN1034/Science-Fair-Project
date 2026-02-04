#!/usr/bin/env python3
"""
phase_locked_stimulation_sim.py - M4 Mac Optimized Version
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import sklearn 
from sklearn.linear_model import LinearRegression
import keras
from keras import layers, models

# --- M4 MAC COMPATIBILITY SECTION ---
import os
import sys

try:
    import tensorflow as tf
    # Fix for the 'dlopen' error: Manually add the plugin path to the system search
    plugin_path = os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages', 'tensorflow-plugins')
    os.environ['DYLD_LIBRARY_PATH'] = plugin_path + ":" + os.environ.get('DYLD_LIBRARY_PATH', '')

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ M4 GPU Detected: {gpu_devices}")
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    
    # Modern Keras 3 import
    import keras
    from keras import layers, models
    HAS_TF = True
except Exception as e:
    print(f"⚠️ TF Loading Error: {e}")
    HAS_TF = False
# -------------------------------------

FEATURES = "emg_imu_features.csv"
STIM_PHASE_OFFSET = pi/2  # 180deg phase shift (destructive) — conceptual only
PULSE_WIDTH_MS = 20     # simulated pulse width
MAX_PULSES = 50

# Safety note (printed)
print("SIMULATION-ONLY: Do NOT apply pulses to humans from this script. Test on resistor loads only.")

# ---------- load features ----------
# Note: Ensure this file is in the same folder as your script
try:
    df = pd.read_csv(FEATURES)
except FileNotFoundError:
    print(f"Error: Could not find {FEATURES}. Please ensure the file exists.")
    exit()

t = df["time_s"].values
amp = df["IMU_Amplitude"].values
phase = df["IMU_Phase"].values

# ---------- find phase crossing times where to stimulate ----------
def wrap(x):
    return (x + pi) % (2*pi) - pi

inst_phase = np.mod(phase, 2*pi)  # 0..2pi
REF_ANGLE = 0.0
target_angle = np.mod(REF_ANGLE + STIM_PHASE_OFFSET, 2*pi)

# detect times where phase crosses target_angle from below
crossings = []
for i in range(1, len(inst_phase)):
    if inst_phase[i-1] < target_angle <= inst_phase[i]:
        crossings.append(i)

print(f"Detected {len(crossings)} candidate stim times (phase crossings).")

# limit pulses
crossings = crossings[:MAX_PULSES]
stim_times = t[crossings]
print("First stim times (s):", stim_times[:10])

# ---------- simulate effect on amplitude ----------
alpha = 0.6   
tau = 0.5     

sim_amp = amp.copy()
for idx in crossings:
    sim_amp[idx:] *= alpha

# ---------- plot before/after ----------
plt.figure(figsize=(10,5))
plt.plot(t, amp, label="original amplitude")
plt.plot(t, sim_amp, label="after simulated pulses")
plt.scatter(stim_times, sim_amp[crossings], color='red', label="stim times")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Simulated phase-locked stimulation effect (toy model)")
plt.show()

# ---------- simple MPC baseline (predictive suppression) ----------
A = amp[:-1].reshape(-1,1)
Y = amp[1:]
model_lr = LinearRegression().fit(A, Y)
a_hat = model_lr.coef_[0]
print("AR(1) fit a_hat:", a_hat)

b_hat = -0.5
lam = 0.01
def mpc_control_step(amp_t):
    return -b_hat*(a_hat*amp_t) / (b_hat**2 + lam)

u_seq = np.zeros_like(amp)
sim_amp_mpc = amp.copy()
for i in range(len(amp)-1):
    u = mpc_control_step(sim_amp_mpc[i])
    u_seq[i] = u
    sim_amp_mpc[i+1] = a_hat*sim_amp_mpc[i] + b_hat*u

plt.figure(figsize=(10,4))
plt.plot(t, amp, label="original")
plt.plot(t, sim_amp_mpc, label="MPC-simulated")
plt.legend()
plt.title("Naive MPC simulation (toy linear model)")
plt.show()

# ---------- LSTM prediction toy (M4 Optimized) ----------
if HAS_TF:
    try:
        print("TensorFlow detected; running M4-accelerated LSTM.")
        fs = 1.0/np.median(np.diff(t))
        horizon = int(0.1*fs)  
        seq_len = int(0.5*fs)  
        X, Y_lstm = [], []
        for i in range(len(amp)-seq_len-horizon):
            X.append(amp[i:i+seq_len])
            Y_lstm.append(amp[i+seq_len+horizon])
        X = np.array(X)[:,:,None]
        Y_lstm = np.array(Y_lstm)
        
        # small LSTM
        model_lstm = models.Sequential([
            layers.Input(shape=(seq_len,1)),
            layers.LSTM(32),
            layers.Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        
        # Slightly larger batch size to utilize M4 bandwidth
        model_lstm.fit(X, Y_lstm, epochs=5, batch_size=128, verbose=1)
        print("LSTM trained. Use model_lstm.predict for horizon forecasts.")
    except Exception as e:
        print("Error during LSTM training:", e)
else:
    print("Skipping LSTM (TensorFlow/Keras not properly installed).")
# ---------- Test the Prediction ----------
test_idx = 100  # Pick a random spot in the data
sample_input = X[test_idx].reshape(1, seq_len, 1)
true_value = Y_lstm[test_idx]
predicted_value = model_lstm.predict(sample_input, verbose=0)

print(f"\n--- M4 Prediction Test ---")
print(f"Actual Amplitude:    {true_value:.6f}")
print(f"Predicted Amplitude: {predicted_value[0][0]:.6f}")
print(f"Difference:          {abs(true_value - predicted_value[0][0]):.6e}")
# End of simulation script