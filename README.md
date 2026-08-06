# BATPING — Bidirectional Acoustics Traveltime Pulser for INdex Gradients

Acoustic thermometry system for characterizing temperature fluctuations inside the Rubin Observatory telescope enclosure. Measures air temperature by tracking round-trip ultrasonic time-of-flight (TOF); temperature sensitivity comes from dc/dT ≈ 0.6 m/s/°C, making 1 μs TOF precision equivalent to ~0.1°C.

**Lab notebook / project log:** [Confluence page](https://harvardwiki.atlassian.net/wiki/spaces/hufasstubbsgroup/pages/878510090)

---

## Repository layout

```
siglent_project/          ← main BATPING pipeline (active)
  chirp_stubbs_analysis.py    core dual-template analysis pipeline
  chunk_gui.py                interactive 3-panel diagnostic GUI
  picoscope_batping_record.py PicoScope recording script (Ch A audio, Ch B TMP36)
  stubbs_longterm_xcorr_vtestv1.py  long-recording batch analysis (current best)
  chirp_sweep.py / alpha_sweep.py   waveform parameter sweeps
  ecm999_full_sweep.py / ecm999_stepped_sine.py  mic transfer function
  voltage_calibration.py      SDG output level calibration
  tmp36_calibration.py        TMP36 sensor verification on PicoScope Ch B
  run_lsfit_stats.py          compute σ statistics from a batch of runs
  analyze_statistics_campaign.py   aggregate statistics across recordings
  run_oscillation_heatpad.sh  timed heatpad ON/OFF protocol with notifications
  recordings/                 per-run analysis results (see recordings/README.md)

agilent_project/          ← earlier Agilent AWG work (archived)
  bayesian_optimization/      Bayesian waveform optimization (June 2026)

accel_magneto_v2.py       ← Raspberry Pi Sense HAT pointing (separate subproject)
```

---

## Hardware

| Component | Model | Notes |
|---|---|---|
| AWG | Siglent SDG1062X | LAN at 10.11.13.230:5025 |
| DAQ | PicoScope 5000D | 384 kSa/s Ch A (audio) + Ch B (TMP36) |
| Transmitter | 40 kHz piezo array | Murata MA40S4S-R, Q~20, BW~4 kHz |
| Microphone | Dodotronic UltraMic 384k EVO | switch 1 (−33 dB gain) |
| Power amp | Siglent SPA1010 | used for 30V drive tests |
| Temp sensor | TMP36 | 10 mV/°C, placed in beam center |

**Geometry:** piezo array + mic side by side, 1 m to flat reflector, 2 m round-trip. TOF ≈ 5.8 ms.

---

## Waveform — selected chirp `a0.3_sm_4ms`

Linear frequency-modulated chirp, finalized after waveform sweeps in July 2026:

| Parameter | Value |
|---|---|
| f_lo / f_hi | 38,450 / 44,450 Hz |
| Bandwidth | 6 kHz |
| Duration | 4 ms |
| Rep rate | 52.632 Hz (~19 ms period) |
| Amplitude factor α | 0.3 (partial H(f)⁻¹ compensation) |
| Conversion | 10.3 μs/°C at 20°C |

---

## Analysis pipeline — Stubbs vtestv1

1. **Record** with `picoscope_batping_record.py` → RF64 WAV (Ch A) + TMP36 CSV (Ch B)
2. **Chunk** into 30 s segments; first 100 periods co-added to build T_launch / T_return templates
3. **Per-period cross-correlation** with shape-fit (LS sinc model, ±700 μs search radius)
4. **Δτ = τ_return − τ_launch** per ping → temperature change via c(T)
5. **Run** `stubbs_longterm_xcorr_vtestv1.py` on a recording directory

Key fixes in vtestv1 vs. earlier versions:
- ±700 μs search radius (catches slow drift ≥ 200 μs from baseline)
- Minimum burst window = BURST_MS (prevents template truncation on low-amplitude bursts)
- Incident right edge capped at `ref_onset − 0.5 ms` (not 0.4×TOF, which cut off the burst for large TOF)

---

## Noise floor results (Aug 2026)

| Condition | σ (μs) | °C/ping |
|---|---|---|
| Open, no fan (enc. stats campaign, N=10) | 0.427 | 0.041 |
| Enclosure, no fan (N=10) | 0.803 | 0.078 |
| Open, fan transverse (N=12, 20-min) | 1.71 (median) | 0.166 |

Fan turbulence dominates at ~4× the open baseline. The slow correlated drift (5–10 s timescale) is real acoustic refractive-index noise, not pipeline artifact.

---

## Dependencies

```
numpy scipy matplotlib soundfile
picosdk (PicoScope 5000 series SDK)
pyvisa (SDG1062X via LAN/TCP)
```
