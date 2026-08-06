# BATPING Data Log

This document tracks all measurements and their provenance to ensure consistency across analyses.

---

## Measurement Sessions

### Session 1: May 26, 2026 ~12:10-12:26 (3rd Floor Office)
**Location:** Third floor office (high 40kHz HVAC interference)  
**Hardware:** Agilent 33220A, Ultramic 384K, 40kHz piezo array  
**Separation:** ~22 cm  
**Drive amplitude:** 5 Vpp  

**Issue:** High 40kHz ambient interference from HVAC made measurements unreliable. Session abandoned after noise characterization.

**Files:**
- `calibration_recording.png`
- `bandpass_comparison.png`
- `fwhm_analysis.png`
- `ambient_ultrasonic.png`, `ambient_no_transducer.png`, `ambient_no_agilent.png`, `ambient_mic_moved.png`

---

### Session 2: May 26, 2026 ~13:34-13:57 (2nd Floor Office - Run 1)
**Location:** Second floor office (low 40kHz interference, some 33kHz motion sensor)  
**Hardware:** Agilent 33220A, Ultramic 384K, 40kHz piezo array  
**Separation:** 18.7 cm (corrected from initial estimate of ~22cm)  
**Drive amplitude:** 5 Vpp  
**Waveform points:** 4000  
**Effective sample rate:** 1 MSa/s  
**Rep rate:** 30 Hz (derived from 4000 pts / 1 MSa/s = 4ms period, but actual output may differ)

**Optimization Run 1:**
- Iterations: 129
- Best iteration: 85
- **Baseline amplitude:** 0.0722 (Gaussian σ=5 cycles)
- **Optimized amplitude:** 0.6730
- **Baseline FWHM:** 354 μs
- **Optimized FWHM:** 396 μs

**Files:**
- `optimized_params.npy` - Best Fourier coefficients from Run 1
- `optimization_results.png`, `optimization_results_v2.png`
- `location_comparison.png`
- `clean_pulse_test.png`, `clean_pulse_test_v2.png`, `high_drive_test.png`

---

### Session 3: May 26, 2026 ~14:51-15:09 (2nd Floor Office - Run 2)
**Location:** Second floor office  
**Hardware:** Same as Session 2  
**Separation:** 18.7 cm  
**Drive amplitude:** 5 Vpp  

**Optimization Run 2:**
- Iterations: 142
- Best iteration: 142
- **Baseline amplitude:** ~0.14 (measured during optimization run)
- **Optimized amplitude:** 0.5712
- **Baseline FWHM:** 279 μs
- **Optimized FWHM:** 385 μs

**Re-recording for comparison plots (separate from optimization loop):**
- **Unoptimized amplitude:** 0.0595
- **Optimized amplitude:** 0.0631
- **Both FWHM:** 357 μs

**Files:**
- `optimized_params_run2.npy` - Best Fourier coefficients: [1.003, 0.306, ~0, ~0, ~0, ~0, ~0, ~0]
- `optimization_results_run2.png`
- `comparison_unoptimized_run2.png`, `comparison_optimized_run2.png`

---

## Amplitude Discrepancy Analysis

| Source | Unoptimized Amp | Optimized Amp | Ratio |
|--------|-----------------|---------------|-------|
| Run 1 optimization loop | 0.0722 | 0.6730 | 9.3x |
| Run 2 optimization loop | ~0.14 | 0.5712 | ~4x |
| Run 2 re-recording | 0.0595 | 0.0631 | 1.06x |
| Session 4 (16:23) | 1.3663 | 1.3479 | 0.99x |

**Key observation 1:** The same waveforms measured in the optimization loop vs. a separate re-recording session show vastly different results. This indicates significant **session-to-session variability** that needs investigation.

**Key observation 2 (Session 4):** When properly recorded with sufficient settling time and recording duration, both waveforms produce **nearly identical peak amplitudes** (~1.35). The apparent "optimization" improvement during the optimization loop was an artifact.

**Possible causes:**
1. Microphone AGC (automatic gain control) responding to ambient sounds
2. Acoustic coupling variations (mic/transducer alignment)
3. Temperature effects on sound speed and transducer response
4. USB timing/buffer issues affecting recording alignment with pulses
5. AWG output state not fully stabilized

**Action items:**
- [ ] Verify Ultramic AGC settings
- [ ] Implement reference amplitude tracking between sessions
- [ ] Log environmental conditions (temperature, ambient noise level)

---

### Session 4: May 26, 2026 16:23 (Controlled Comparison)
**Location:** Second floor office  
**Hardware:** Agilent 33220A, Ultramic 384K, 40kHz piezo array  
**Separation:** 18.7 cm  
**Drive amplitude:** 5 Vpp  
**Settling time:** 200 ms  
**Recording duration:** 150 ms  

**Results:**
- **Unoptimized peak:** 1.3663, FWHM: 547 μs
- **Optimized peak:** 1.3479, FWHM: N/A (continuous)
- **Ratio:** 0.99x (essentially identical)

**Critical finding:** The "optimized" waveform produces a **continuous signal**, not discrete pulses:
- Unoptimized waveform: 10.8% duty cycle → clear discrete pulses
- Optimized waveform: 91.7% duty cycle → essentially continuous

The optimizer converged to a near-continuous waveform because:
1. Waveform duration (4ms) equals the repetition period (250 Hz = 4ms)
2. Maximizing energy transfer to resonant piezo favors sustained excitation
3. The objective function (maximize amplitude, minimize FWHM) doesn't penalize continuous signals

**Implication:** The optimized waveform is **unsuitable for TOF measurements** because there's no discrete pulse arrival to time. The optimization objective needs to be reformulated to enforce pulsed behavior.

**Files:**
- `comparison_final_20260526.png` - Full comparison with corrected separation
- `comparison_combined_20260526_162357.png` - Earlier version
- `comparison_data_20260526_162357.npz` - Raw audio data
- `back_to_back_test.png` - Debug plot showing intermittent recording issues

---

## Standard Reference Waveforms

For consistent comparison across sessions, use these definitions:

### Unoptimized Baseline
- **Type:** Gaussian envelope
- **Carrier:** 40 kHz
- **Envelope σ:** 5 carrier cycles (125 μs)
- **Waveform points:** 4000
- **Sample rate:** 1 MSa/s
- **Total duration:** 4 ms
- **Active region:** ~1.74 - 2.26 ms (0.52 ms FWHM of envelope)

### Optimized (Run 2)
- **Type:** Fourier-parameterized envelope
- **Coefficients:** a₀=1.003, a₁=0.306, others ≈ 0
- **Envelope formula:** `env(t) ≈ 1.0 + 0.3·cos(2πt/T)`
- **Carrier:** 40 kHz
- **Waveform points:** 4000
- **Sample rate:** 1 MSa/s
- **Total duration:** 4 ms
- **Active region:** 0 - 4 ms (essentially continuous)

---

## Plot Annotation Requirements

All future plots should include:
1. **Date/time** of data acquisition
2. **Session identifier** (e.g., "Run 2")
3. **Key parameters:** drive amplitude, separation distance
4. **Measured values** with units
5. **Waveform type** if showing input (Gaussian/Fourier/etc.)

---

## Analysis: Why the Optimizer Converged to Continuous Waveform

### The Problem

The Nelder-Mead optimizer with objective function:
```
score = amplitude_weight × peak_amplitude - width_weight × (FWHM / 100)
```
converged to Fourier coefficients [a₀≈1.0, a₁≈0.3, others≈0] which produces an envelope:
```
env(t) ≈ 1.0 + 0.3·cos(2πt/T)
```

This results in a **91.7% duty cycle** waveform — essentially continuous 40kHz output.

### Why This Happened

1. **Waveform period = Rep period:** With 4000 points at 1 MSa/s and 250 Hz rep rate, the waveform fills the entire period with no gap.

2. **Resonant piezo favors continuous excitation:** The 40kHz piezo transducer is a resonant system. Sustained driving allows mechanical amplitude to build up, maximizing acoustic output.

3. **FWHM metric is irrelevant for continuous signals:** When the signal never drops below half-max, FWHM becomes undefined or equals the full recording length. The width penalty doesn't constrain the solution.

4. **No duty-cycle constraint:** The objective function doesn't penalize high duty cycles or enforce pulse-like behavior.

### Measured Reality (Session 4)

| Waveform | Peak Amplitude | FWHM | Duty Cycle |
|----------|---------------|------|------------|
| Gaussian (σ=5) | 1.366 | 547 μs | 10.8% |
| Optimized | 1.348 | N/A | 91.7% |

**Both produce identical peak amplitudes** when the piezo reaches steady state. The optimizer's apparent 4-9x improvements during optimization runs were artifacts of:
- Insufficient settling time before recording
- Recording timing not aligned with pulse arrivals
- Run-to-run variability

### Implications for TOF Measurements

A continuous 40kHz signal **cannot be used for time-of-flight measurements** because:
- No discrete pulse arrival to detect
- Phase information is ambiguous over long distances
- Cannot distinguish direct path from reflections

### Recommendations

1. **Reformulate objective:** Add explicit constraints:
   - Maximum duty cycle (e.g., <30%)
   - Minimum inter-pulse silence
   - Penalize FWHM relative to waveform period, not absolute

2. **Use shorter waveform period:** If waveform is 1ms but rep period is 4ms, there's forced silence.

3. **Optimize for arrival time precision:** Instead of peak amplitude, optimize for:
   - Rise time (10-90%)
   - Threshold crossing sharpness
   - Signal-to-noise at arrival detection point

4. **Consider two-stage approach:**
   - Stage 1: Optimize for amplitude with pulsed constraint
   - Stage 2: Minimize width while maintaining amplitude

---

*Last updated: May 26, 2026*
