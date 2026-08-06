# recordings/

Each subdirectory is one acquisition run, named `YYYYMMDD_HHMMSS_<label>/`.

Raw WAV and AIFF files are gitignored (too large). What is tracked per recording:
- Summary NPZ and PNG at the top level of each run directory
- `analysis_xcorr_vtestv1/` — vtestv1 analysis results (NPZ + summary PNG)
- TMP36 CSV if applicable

Per-chunk analysis PNGs (`stubbs_pipeline_vtestv1/stubbs_chunk_*.png`) are **not tracked** in git — they are regenerated locally by running the analysis script.

---

## Recording categories

### Waveform sweeps (June–July 2026)
Early characterization runs. Full-sweep plots and manifests. Barker-7 sweeps removed from git; data archived locally.

| Directory | Description |
|---|---|
| `20260705_*_chirp_sweep/` | LFM bandwidth/duration sweeps |
| `20260708_133617_alpha_sweep/` | α-compensation factor sweep |
| `20260710_031308_alpha_sweep_voltage_compensated/` | α-sweep with SDG voltage calibration |
| `20260715_161527_chirp_sweep_V2_newHf/` | Chirp sweep with updated ECM999 H(f) |
| `20260715_162235_alpha_sweep_v2_newHf/` | α-sweep with updated H(f) |
| `20260710_194120_longterm_a0.3_sm_4ms/` | First long-term recording, selected waveform |

### Mic / transducer characterization
| Directory | Description |
|---|---|
| `20260715_154756_sanity_check_ultramic_v2_newHf/` | UltraMic response sanity check |
| `20260716_015251_ecm999_full_sweep_v3_ecmHf/` | ECM999 stepped-sine transfer function |
| `20260725_150511_ecm999_full_sweep_v3_ecmHf/` | Final ECM999 H(f) measurement |

### Heat tests — dummy load on transducer board (July 2026)
Tests heating the amplifier board without acoustic heatpad. **Baseline**: open air, no fan.

| Directory | Notes |
|---|---|
| `20260717_170007_batping_heattest/` through `20260717_173325_*` | First heat test (18W dummy load). Failed: turbulence + wrong sample rate. |
| `20260725_163650_batping_heattest_20V/` | 20V tests (before enclosure) |
| `20260725_183814_batping_heattest_30V_take1/` | 30V take 1 (open) |
| `20260727_105957_batping_heattest_30V_take2/` | 30V take 2 (open) |
| `20260727_120102_batping_heattest_30V_take3/` | 30V take 3 (open) |
| `20260728_032243_batping_heattest_30V_take4/` | **30V take 4 (enclosed, SenseHat)** — best heat test |
| `20260728_040447_batping_heattest_30V_take4b/` | Take 4b continuation |
| `20260728_214103_batping_heattest_flat3db_12p5V/` | 12.5V flat-3dB waveform test |

### Heatpad CVS tests (Aug 2026)
Heatpad placed in acoustic beam center. TMP36 sensor also in beam from take3 onward.

| Directory | Notes |
|---|---|
| `20260801_104613_batping_heatpad_CVS_open_take1/` | Open, no enclosure |
| `20260801_113036_batping_heatpad_CVS_open_take2/` | Open, recording interrupted at 29.25 min |
| `20260801_123228_batping_heatpad_CVS_enclosure_take1/` | Enclosed |
| `20260801_133536_batping_heatpad_CVS_enclosure_take2/` | Enclosed |
| `20260801_150236_batping_heatpad_CVS_enclosure_take3_tmp36/` | Enclosed + TMP36 |
| `20260801_162747_batping_heatpad_CVS_enclosure_oscillate_take1/` | Oscillation (OFF/HIGH cycles) |

### Enclosure noise-floor baselines (Aug 2026)
30 s recordings, enclosed, no heatpad.

| Directory | Notes |
|---|---|
| `20260730enclosure_statistics_test/` | N=10 open + N=10 enclosure statistics campaign |
| `20260801_174318_*` through `20260801_194545_*` | N=5 enclosed baselines with TMP36 |

### Open fan baselines (Aug 2026)
20-min recordings, open air, desk fan transverse to acoustic path. N=12 recordings.

| Directory | σ_vtestv1 (μs) |
|---|---|
| `20260802_164348_batping_baseline_open_fan/` | 1.16 |
| `20260802_170440_batping_baseline_open_fan/` | 1.06 |
| `20260802_172529_batping_baseline_open_fan/` | 2.35 |
| `20260802_174621_batping_baseline_open_fan/` | 2.77 |
| `20260802_180715_batping_baseline_open_fan/` | 1.68 |
| `20260802_182808_batping_baseline_open_fan/` | 3.78 |
| `20260802_184902_batping_baseline_open_fan/` | 1.73 |
| `20260802_190958_batping_baseline_open_fan/` | 1.68 |
| `20260802_193052_batping_baseline_open_fan/` | 1.79 |
| `20260802_195143_batping_baseline_open_fan/` | 1.15 |
| `20260802_201237_batping_baseline_open_fan/` | 1.21 |
| `20260802_203333_batping_baseline_open_fan/` | 1.96 |

Median σ = 1.71 μs (~4× open baseline without fan). Dominated by real turbulence-induced slow drift, not pipeline noise.

---

## SenseHat / auxiliary data

| File | Description |
|---|---|
| `heattest_30V_sensehat.csv` | SenseHat temperature during 30V heat tests |
| `heattest_30V_sensehat_cooldown.csv` | Cooldown phase |
| `heattest_20V_sensehat.csv` | 20V heat test temperature |
| `heattest_diagnostic.png` | Δτ vs SenseHat temperature overlay |
| `README_heattest_20260717.md` | Full procedure for July 17 dummy-load heat test |
