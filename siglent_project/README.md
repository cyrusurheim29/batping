# siglent_project — BATPING Analysis Pipeline

This directory contains all scripts and results for the BATPING acoustic thermometry system. Hardware is driven by a Siglent SDG1062X AWG and recorded by a PicoScope 5000D.

---

## Core scripts

### Recording
| Script | Purpose |
|---|---|
| `picoscope_batping_record.py` | Main recording script. Streams Ch A (audio, 384 kSa/s) + Ch B (TMP36 temperature). Writes RF64 WAV + TMP36 CSV. Includes SDG keepalive to prevent TCP timeout. |
| `voltage_calibration.py` | Measure SDG output voltage and calibrate level before heat tests. |
| `tmp36_calibration.py` | Quick TMP36 verification on PicoScope Ch B (no SDG required). |

### Analysis
| Script | Purpose |
|---|---|
| `chirp_stubbs_analysis.py` | **Core pipeline.** Dual-template co-add → per-period xcorr shape-fit → Δτ time series. All key analysis parameters live here (BURST_MS, search radius, window guards). |
| `stubbs_longterm_xcorr_vtestv1.py` | **Batch long-recording analysis (current best).** Wraps `chirp_stubbs_analysis.py` for 20–40 min recordings; outputs to `analysis_xcorr_vtestv1/`. |
| `stubbs_longterm_xcorr_vtest.py` | Older batch script (±700 μs search, no minimum window fix). |
| `chunk_gui.py` | Interactive 3-panel diagnostic GUI. Loads one chunk, shows templates + xcorr + Δτ. Run as `python chunk_gui.py <chunk.wav>`. |
| `run_lsfit_stats.py` | Compute σ and median Δτ from a batch of vtestv1 NPZ output files. |
| `analyze_statistics_campaign.py` | Aggregate noise floor statistics across an ensemble of 30 s recordings. |

### Sweeps / Characterization
| Script | Purpose |
|---|---|
| `chirp_sweep.py` | LFM chirp parameter sweep (BW, duration, compensation factor). |
| `alpha_sweep.py` | α-factor sweep over H(f)⁻ᵅ compensation (α = 0.2–0.8, sm/md/lg BW). |
| `ecm999_full_sweep.py` | Full stepped-sine transfer function of ECM999 microphone. |
| `ecm999_stepped_sine.py` | Single-frequency stepped-sine burst characterization. |
| `piezo_characterization.py` | Chirp-based piezo bandwidth measurement. |
| `ultramic_transfer_function.py` | Transfer function measurement for UltraMic. |
| `barker7_prbs_testing.py` | Barker-7 BPSK coded waveform sweep (archived; chirp outperforms). |

### Utilities
| Script | Purpose |
|---|---|
| `batping_timeseries_viewer.py` | Quick viewer for Δτ time series NPZ files. |
| `run_oscillation_heatpad.sh` | Timed heatpad oscillation protocol with macOS notifications. |
| `run_statistics_campaign.sh` | Batch-run multiple 30 s recordings for noise floor statistics. |

---

## Key result files

| File | Description |
|---|---|
| `statistics_campaign_summary.png` | Noise floor summary across open / enclosure conditions |
| `waterfall_all_heattests.png` | Δτ waterfall across all heat tests and baselines (4°C offset/trace) |
| `ecm999_transfer_function.png/.npz` | ECM999 mic frequency response |
| `stepped_sine_characterization_burst8ms_20260715_152456.png/.npz` | Final stepped-sine piezo H(f) |
| `ultramic_transfer_function_20260715_113049_fixed.png/.npz` | UltraMic H(f) |
| `piezo_characterization.png/.npz` | Chirp-based piezo bandwidth |
| `barker7_cycles_sweep_summary_v6.png` | Barker-7 FWHM vs cycles/chip sweep summary |
| `waveform_comparison_table_20260707_102902.png` | σ_TOF comparison across all tested waveforms |
| `theo_vs_meas_sigma_tof.png` | Cramér-Rao bound vs measured σ_TOF |
| `poster_schematic.png` | Hardware schematic for poster/presentations |

---

## Pipeline parameter reference

These live at the top of `chirp_stubbs_analysis.py`:

```python
F_LO_HZ   = 38450    # chirp start frequency
F_HI_HZ   = 44450    # chirp end frequency
BURST_MS  = 4.0      # minimum burst window width (vtestv1 fix)
N_TEMPLATE = 100     # periods co-added for templates
REP_RATE_HZ = 52.632 # SDG repetition rate
```

Search radius in `stubbs_longterm_xcorr_vtestv1.py`: `_search_radius_us = 700.0`

---

## Recordings directory

See `recordings/README.md` for a guide to the recording structure.

Recordings of interest (all in `recordings/`):
- `20260728_032243_batping_heattest_30V_take4/` — best-quality heat test (enclosed, 30V)
- `20260801_150236_batping_heatpad_CVS_enclosure_take3_tmp36/` — heatpad + TMP36 overlay
- `20260801_162747_batping_heatpad_CVS_enclosure_oscillate_take1/` — oscillation test
- `20260802_164348_batping_baseline_open_fan/` through `20260802_203333_*` — 12× fan baseline

---

## Documentation

- `sigma_tof_derivation.tex` — Cramér-Rao derivation of σ_TOF from FWHM and SNR
- `prbs_waveform_theory.tex` — Barker/MLS theory
- `README_piezo_characterization.md` — detailed characterization methodology
- `advisor_report_20260720.tex` — mid-summer progress report
