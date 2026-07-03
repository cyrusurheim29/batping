# BATPING Siglent Project

Ultrasonic time-of-flight (TOF) measurement system for acoustic thermometry.

## Goal

Measure air temperature via speed of sound: c(T) = 331.3 + 0.6T m/s. For a 2-meter round-trip path, achieving 0.1°C precision requires TOF precision < 1 μs.

## Hardware

- **AWG**: Siglent SDG1062X via LAN (10.11.13.230:5025)
- **Microphone**: Dodotronic UltraMic @ 384 kHz
- **Transducers**: 40 kHz piezoelectric array (resonant, Q~20)
- **Geometry**: Piezo + mic side-by-side, 1 m to flat reflector (2 m round-trip)

## Piezo Characterization (June 2026)

Measured via chirp method (see `README_piezo_characterization.md`):
- **Center frequency**: 41.0 kHz
- **-3 dB bandwidth**: 2.0 kHz (passband 40.0 - 42.0 kHz)
- **-6 dB bandwidth**: 4.8 kHz
- **Q factor**: ~20

## Waveform Optimization

### Approach
Bayesian optimization over parameterized Gaussian pulse with pre/post compensation bursts. Objective: minimize response envelope autocorrelation FWHM (determines TOF precision).

### Results

| Version | Iterations | σ (cycles) | Autocorr FWHM | Improvement |
|---------|------------|------------|---------------|-------------|
| Baseline | - | 5.0 | 443 μs | - |
| V1 | 100 | ~3 | ~350 μs | ~20% |
| V2 | 200 | 2.32 | 307 μs | 31% |
| V3 | 1000 | 1.93 | 275 μs | 38% |

### Key Finding
Optimizer consistently converges to σ ≈ 2 cycles regardless of iterations, suggesting this is near the hardware-limited optimum given the 2 kHz piezo bandwidth. Theoretical floor is ~250 μs (≈ 1/bandwidth).

## Next Steps: PRBS Waveforms

Pseudo-random binary sequences (Barker codes, MLS) could achieve sharper correlation peaks through pulse compression. See `prbs_waveform_theory.tex` for theory.

**Constraints from piezo bandwidth:**
- Chip duration T_c ≥ 1/B = 500 μs (aggressive) to 1000 μs (conservative)
- Max ~10-12 chips before multipath (5-6 ms window)
- Barker-7, Barker-11, or Barker-13 are candidates

**Pending**: ~~Advisor suggests stepped-sine characterization to validate chirp-based bandwidth measurement.~~ Implemented — see `stepped_sine_characterization.py`.

## Stepped-Sine Validation (June 15, 2026)

Validates chirp-based bandwidth measurement by sweeping single-frequency sines across the band:
- **Frequency range**: 25-55 kHz at 25 Hz steps (1200 measurements)
- **Method**: For each frequency, drive piezo with pure sine, record 0.3s, extract |H(f)| via single-bin FFT
- **Runtime**: ~25 minutes
- **Output**: `stepped_sine_characterization.npz`, standalone plot, and comparison overlay with chirp result

Run with:
```bash
python stepped_sine_characterization.py
```

## Barker-7 BPSK Testing Version History

### V1 (June 16, 2026)
- Initial implementation with broken SDG upload (40,000 samples exceeded 16,384 limit)
- Waveform played at wrong rate, carrier compressed to ~400 Hz

### V2 (June 16, 2026)
- Fixed sample count to stay under SDG limit
- `rep_rate = SDG_SAMPLE_RATE / len(waveform)` preserves carrier frequency
- **Problem**: No gap between pulses — piezo continuously driven, causing blended correlation peaks
- TOF inconsistent (1940–9229 μs), FWHM stuck at 10.4 μs (measuring carrier oscillations)

### V3 (June 17, 2026)
- Added inter-pulse spacing: 40 Hz rep rate, ~21 ms dead time for piezo settling
- Waveform embedded in 10,000-sample buffer with trailing zeros

### V4 (June 17, 2026)
- **Autocorrelation-based measurement**: Replaced matched-filter cross-correlation with received signal autocorrelation
  - Previous approach: Cross-correlate received signal with theoretical waveform template
  - Problem: Template mismatch (theoretical ≠ what piezo actually emits), arbitrary TOF values
  - V4 approach: Autocorrelate the received signal itself
  
- **Why autocorrelation is better**:
  1. No template mismatch — measures what we actually received
  2. TOF emerges naturally from peak spacing:
     - Main peak at lag=0 (signal correlated with itself)
     - Secondary peak at lag=TOF (incident pulse ⊗ reflected pulse)
  3. FWHM from main peak = true achievable resolution

- **Autocorrelation structure**:
  ```
  Received signal contains: incident pulse (t≈0) + reflected pulse (t≈TOF)
  
  Autocorrelation peaks:
    lag = 0      → main peak (largest) → measure FWHM here
    lag = ±TOF   → incident ⊗ reflected cross-terms → TOF measurement
    lag = ±25ms  → consecutive pulses (40 Hz rep rate)
  ```

- **Envelope detection**: Hilbert transform extracts envelope before measuring FWHM/sidelobes
  - V2/V3 bug: measured carrier oscillations (~10 μs) instead of correlation envelope

- **Theoretical vs Measured metrics**: Reports both
  - *Theoretical*: Autocorrelation of computer-generated waveform (ideal pulse compression)
  - *Measured*: Autocorrelation of received signal (actual achieved resolution)

- Expected results:
  - Theoretical FWHM: ~1 chip duration (387–1161 μs for 16–48 cyc/chip)
  - Measured FWHM: Wider due to piezo bandwidth limitation
  - Theoretical sidelobes: ~-16.9 dB (Barker-7 property)
  - TOF: ~5800 μs for 2m round-trip (consistent across all cycles/chip)

- **FWHM measurement caveat**: Currently measured at lag=0 (main peak), not at lag=TOF.
  - Lag=0 FWHM reflects intrinsic waveform resolution
  - For coded waveforms (Barker, MLS), sidelobes from lag=0 peak can overlap with TOF peak
  - This interference could degrade actual TOF measurement precision beyond what lag=0 FWHM suggests
  - Future work: compare lag=0 FWHM vs TOF peak FWHM to quantify this effect

### V5 (June 18, 2026)
- **Sidelobe fix**: Now calculated relative to main peak (not absolute dB)
  ```python
  # V4 (wrong): absolute dB
  peak_sidelobe_db = 20 * log10(peak_sidelobe)
  
  # V5 (correct): relative to main peak
  peak_sidelobe_db = 20 * log10(peak_sidelobe / main_peak)
  ```
  Expected sidelobe levels: Barker-7 = -16.9 dB, Barker-11 = -20.8 dB, Barker-13 = -22.3 dB

- **Transfer efficiency fix**: Normalized by signal length
  ```python
  # V4 (wrong): compared autocorr peaks of different-length signals
  efficiency = max(rx_autocorr) / max(theo_autocorr)  # Meaningless!
  
  # V5 (correct): normalize by signal length first
  theo_peak_norm = max(theo_autocorr) / len(theo_waveform)
  rx_peak_norm = max(rx_autocorr) / len(received)
  efficiency = rx_peak_norm / theo_peak_norm
  ```

- **Data archival system**: Structured output for offline analysis
  ```
  recordings/YYYYMMDD_HHMMSS_barker7_sweep/
  ├── manifest.json    # Run metadata, hardware config, all results
  ├── raw/*.wav        # Received signals at mic sample rate (384 kHz)
  ├── plots/*.png      # Individual and summary plots
  └── results.npz      # Numerical results for programmatic access
  ```

- **manifest.json** contains: timestamp, version, hardware config, test parameters,
  and per-measurement results (FWHM, sidelobe, TOF, efficiency) with file paths

## Waveform Comparison (July 2026)

TOF-peak FWHM from full-signal autocorrelation, HP 30 kHz order-8 filter.
Metric: FWHM of the incident⊗reflected cross-correlation peak at lag=TOF.

| Waveform | TOF FWHM (μs) | Notes |
|----------|---------------|-------|
| **LFM 8 kHz comp (37–45 kHz, 10 ms)** | **169** | **New best — beats V3** |
| LFM 6 kHz comp (38–44 kHz, 10 ms) | 201 | Piezo-compensated |
| LFM 8 kHz flat (37–45 kHz, 10 ms) | 245 | Uncompensated |
| Optimized Gaussian V3 | 250 | Bayesian-optimized, σ≈2 cycles |
| LFM 12 kHz flat (35–47 kHz, 10 ms) | 232 | Wide flat chirp (same effective BW as 8 kHz) |
| LFM 6 kHz flat (38–44 kHz, 10 ms) | 281 | |
| Barker-7 @ 4 cyc/chip | 253 | Best Barker (unresolved chips, acts like short pulse) |
| Barker-7 @ 16 cyc/chip | 513 | Chips partially resolved |
| Baseline Gaussian (σ=5) | ~479 | Unoptimized reference |

**New theoretical floor: ~125–169 μs** using amplitude-compensated LFM chirp.
Previous floor was ~250 μs (≈ 1/piezo_BW). Compensation extends effective BW to ~6 kHz.

### Chirp Sweep Key Findings (July 2, 2026)

**FWHM depends on bandwidth, not duration.** The 3.75 kHz chirp gives identical TOF FWHM
at 2 ms, 5 ms, and 10 ms (all ~406–443 μs) — the channel is bandwidth-limited, not
energy-limited over this range.

**Amplitude compensation pre-distorts the transmitted envelope by 1/|H(f_inst)| so that
the received spectrum is approximately flat across the chirp bandwidth.** Results for
10 ms chirps at 40 Hz:

| BW | Flat FWHM | Comp FWHM | Improvement |
|----|-----------|-----------|-------------|
| 3.75 kHz | 406 μs | 352 μs | 13% |
| 6 kHz | 281 μs | 201 μs | 28% |
| 8 kHz | 245 μs | **169 μs** | 31% |

**Barker-7 does not outperform Gaussian for this piezo.** Chips shorter than 1/BW (~500 μs)
are not individually resolved — the sequence acts like a single short pulse. Chips longer
than 1/BW create wider TOF peaks than V3 (513 μs at 16 cyc/chip). The smallest TOF FWHM
from Barker (253 μs at 4 cyc/chip) matches V3 but provides no improvement and has
poor sidelobe suppression (~-10 dB measured vs -16.9 dB theoretical).

### Analysis Pipeline Notes

- **analyze_wav_autocorr_v1.py**: envelope must be computed from FILTERED data (fixed July 2026);
  REP_RATE must match the recording (40 Hz for all current-setup recordings, 62.5 Hz for
  older bayesian_optimizer_sdg.py recordings).
- **chirp_sweep.py**: uses full-signal autocorrelation (not matched filter) — the 10 ms
  chirp duration is comparable to TOF (~5.7 ms), making matched-filter peak detection
  unreliable.

## Key Files

### Scripts
- `piezo_characterization.py` - Chirp-based bandwidth measurement
- `stepped_sine_characterization.py` - Stepped-sine bandwidth validation
- `bayesian_optimizer_sdg.py` - Bayesian waveform optimization
- `test_optimized_waveform.py` - Test optimized waveforms

### Data
- `piezo_characterization.npz` - Bandwidth measurement data
- `stepped_sine_characterization.npz` - Stepped-sine piezo response (used for amplitude compensation)
- `optimized_waveform_v2_autocorr_20260611.npz` - V2 optimized Gaussian waveform
- `optimized_waveform_v3_autocorr_20260611.npz` - V3 optimized Gaussian waveform
- `barker7_cycles_sweep_v6.npz` - V6 Barker-7 sweep (full 2–48 cyc, TOF peak FWHM metric)
- `chirp_sweep_results.npz` - Latest chirp sweep summary (not saved yet; see recordings/)
- `recordings/` - Archived runs with WAV files and manifests

### Scripts
- `chirp_sweep.py` - LFM chirp sweep (BW, duration, flat/compensated variants)
- `chirp_reanalyze.py` - Reanalyze saved chirp WAVs with full-signal autocorrelation
- `barker7_prbs_testing.py` - Barker-7 chip sweep (V6: targeted sidelobe search, TOF FWHM)
- `analyze_wav_autocorr_v1.py` - Per-period autocorrelation pipeline for saved WAVs

### Documentation
- `README_piezo_characterization.md` - Detailed characterization notes
- `prbs_waveform_theory.tex` - PRBS/LFSR/Barker theory (~30 pages)
- `waveform_optimization_literature_review.tex` - Literature review

## 25 kHz Noise Investigation (June 30, 2026)

Power spectrum analysis of recorded WAV files revealed a 25 kHz interference spike:

| Recording | 25 kHz Level | 41 kHz Signal | Recording Date |
|-----------|--------------|---------------|----------------|
| optimized_v2_response.wav | -26 dB | -34 dB | June 11 |
| optimized_v3_response.wav | -20 dB | -33 dB | June 11 |
| barker7_24cyc_received.wav | -115 dB | -53 dB | June 18 |

**Findings:**
- The 25 kHz noise is environmental (likely lab equipment), not waveform-dependent
- It was present during June 11 recordings but absent/negligible on June 18
- A 30 kHz highpass filter removes it without affecting the 40 kHz signal

**V3 autocorrelation artifacts:**
The optimized_v3 autocorrelation shows prominent envelope oscillations and elevated noise floor compared to v2. Investigation ongoing — the v3 waveform has a narrower spectrum (6.2 kHz vs 4.4 kHz 3dB bandwidth) which may interact differently with the piezo transfer function.

## Dependencies

```
numpy, scipy, matplotlib
pyaudio (UltraMic recording)
scikit-optimize (Bayesian optimization)
```
