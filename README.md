# Neuroadaptive Closed‑Loop Tremor Modulation (ISEF Project)

This repository contains an ISEF‑level neuroadaptive wearable pipeline for real‑time tremor tracking and modulation using EMG + IMU signals. The system fuses signal processing, CNN+LSTM prediction, hybrid MPC+RL control, and a synthetic dopaminergic feedback model. It is designed to run on a Raspberry Pi 4B with a Teensy microcontroller for sensor acquisition and control messaging.

**Project Title**
A Neuroadaptive Closed‑Loop Wearable Device Integrating Bio‑Inspired Reinforcement Learning and Predictive Control for Real‑Time Non‑Invasive Tremor Modulation and Synthetic Dopaminergic Feedback in Parkinson’s Disease

**What’s Inside**
- Real‑time pipeline for EMG+IMU fusion, prediction, and hybrid control.
- Offline analysis scripts for data preprocessing, model training, and control simulation.
- Logging and evaluation hooks for latency, energy, and tremor suppression metrics.

**Repository Structure**
- `real-time pipeline/realtime.py`  
  Main real‑time pipeline (Pi‑friendly) with DSP, CNN+LSTM inference, MPC, RL, synthetic dopamine, and safety guards.
- `real-time pipeline/config.yaml`  
  Primary configuration for serial, sampling, model, and logging options.
- `non-real time pipeline/`  
  Offline scripts for feature extraction, model training, prediction, and MPC simulation.
- `tflite_env/`, `tflite_312_env/`  
  Local environments for TensorFlow/TFLite workflows.

---

**Quick Start (Raspberry Pi 4B)**
1. Install dependencies
```bash
python3 -m pip install numpy pandas scipy pyserial pyyaml
python3 -m pip install tflite-runtime  # recommended on Pi
```
2. Run the real‑time pipeline
```bash
python3 "real-time pipeline/realtime.py" --config "real-time pipeline/config.yaml"
```
3. Simulation mode (CSV replay)
```bash
python3 "real-time pipeline/realtime.py" --simulate --config "real-time pipeline/config.yaml"
```

---

**Real‑Time Pipeline Overview**
- **Signal Processing**  
  EMG bandpass + envelope, IMU tremor band + analytic phase, PSD and coupling metrics.
- **Prediction (CNN+LSTM)**  
  Uses a trained model (TFLite or Keras) to predict tremor probability, phase, and amplitude.
- **Hybrid Control**  
  Phase‑lock + MPC + RL with safety gating and rate limits.
- **Synthetic Dopamine**  
  Reward prediction error modulates online learning for neuroadaptive behavior.

---

**Models**
- **Single model**: set `model.tflite` or `model.keras_h5` in `real-time pipeline/config.yaml`.
- **Ensemble**: set `model.ensemble` to a list of model paths and optional `model.ensemble_weights`.

Model outputs expected by default:
- `tremor_prob`
- `phase_pred`
- `amp_pred`

These can be remapped using `model.output_map`.

---

**Key Configuration Targets**
These align with your engineering criteria:
- `metrics.latency_target_ms`  
  Control‑loop latency target.
- `metrics.energy_scale`  
  Relative energy estimation per stimulation interval.
- `mpc.*` and `rl.*`  
  Hybrid control tuning knobs.
- `prediction.*`  
  Tremor probability thresholds, phase prediction horizon, and smoothing.

---

**Data Logging**
CSV logs are written to the logs directory:
- Default: `real-time pipeline/logs`  
Each row includes sensor features, model outputs, control actions, dopamine metrics, latency, and energy estimates.

---

**Offline Pipeline (Training & Analysis)**
Use the scripts in `non-real time pipeline/` for:
- EMG/IMU preprocessing
- CNN+LSTM model training and prediction
- MPC simulation and performance logging

Example training script:
```bash
python3 "non-real time pipeline/cnn-lstm_state_estimator.py" --train
```

---

**Safety Checklist**
- Use a USB isolator for any PC‑connected testing.
- Validate stimulation on a resistor load before any human use.
- Keep stimulation amplitude under strict limits and use opto‑isolation hardware.

---

**Troubleshooting**
- `pyserial` errors: confirm serial device path with `ls /dev/ttyACM*`.
- Model not loading: verify `model.tflite` path and that `tflite-runtime` is installed.
- Phase instability: reduce `prediction.phase_prediction_weight` or increase `prediction.smoothing_tau_s`.
- Slow loop: reduce `sampling.window_seconds` or use a smaller TFLite model.

---

**Suggested Next Steps**
1. Tune `mpc` and `rl` parameters using logged data.
2. Convert your trained model to TFLite for real‑time inference on Pi.
3. Add automated experiment modes (baseline vs open‑loop vs closed‑loop).

---

**File Reference**
Main implementation: `real-time pipeline/realtime.py`
