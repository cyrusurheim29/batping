# BATPING Pulse Optimization - Run 1 Summary

**Date:** May 26, 2026  
**Location:** Second floor office (low 40kHz ambient interference)

---

## Objective

Optimize the electrical input waveform driven by the Agilent 33220A to the piezoelectric transducer array such that the acoustic output — as measured by the Dodotronic Ultramic 384K microphone — achieves:
1. **Maximum peak amplitude** (for strong signal-to-noise ratio)
2. **Minimum pulse width (FWHM)** (for precise time-of-flight measurements)

The optimization seeks to find the input waveform that, when driven by the Agilent to the piezo transducer array, produces the peak amplitude/FWHM *as measured by the microphone* that is most optimal (as determined by our objective function), and thus by extension minimizes the effect of the nonlinearity of the piezo transducers which is especially pronounced at their resonant frequency (known otherwise as the ringdown).

---

## Hardware Configuration

| Component | Model | Settings |
|-----------|-------|----------|
| Waveform Generator | Agilent 33220A | USB, Arbitrary mode |
| Transducer | 40kHz Piezoelectric Array | Resonant frequency: 40kHz |
| Microphone | Dodotronic Ultramic 384K EVO | 384kHz sample rate |
| Separation | ~22 cm | Direct path, no reflector |

**Drive parameters:**
- Amplitude: 5 Vpp
- Repetition rate: 30 Hz
- Waveform points: 4000
- Effective sample rate: 1 MSa/s

---

## Optimization Method

### Parameterization
Waveform envelope represented as Fourier series with 8 coefficients:

```
envelope(t) = a₀ + a₁·cos(2πt) + b₁·sin(2πt) + a₂·cos(4πt) + ...
```

The full waveform is: `envelope(t) × sin(2π × 40kHz × t)`

### Objective Function
```
score = amplitude_weight × peak_amplitude - width_weight × (FWHM / 1000)
```
- `amplitude_weight = 1.0`
- `width_weight = 0.3`

### Algorithm
Nelder-Mead simplex optimization (derivative-free, suitable for hardware-in-the-loop)

---

## Results

### Baseline (Unoptimized)
Gaussian envelope with σ = 5 carrier cycles:
- **Peak amplitude:** 0.0722
- **FWHM:** 354 μs

### Optimized Result
After 129 evaluations:
- **Peak amplitude:** 0.6730
- **FWHM:** 396 μs
- **Best iteration:** 85

### Improvement
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Amplitude | 0.0722 | 0.6730 | **+832%** |
| FWHM | 354 μs | 396 μs | +12% |

### Optimized Parameters
```
a₀ = 1.050  (DC component)
a₁ = 0.300  (first cosine harmonic)
a₂ ≈ 0      (higher harmonics negligible)
b₁ ≈ 0
b₂ ≈ 0
...
```

The optimizer converged to a smooth, broad envelope — essentially a tapered continuous wave rather than a compact pulse. This maximizes energy transfer to the piezo at its resonant frequency.

---

## Generated Files

| File | Description |
|------|-------------|
| `optimization_results.png` | 4-panel summary: amplitude history, width history, trade-off scatter, final pulse |
| `comparison_unoptimized.png` | Unoptimized input waveform (top) and acoustic response (bottom) |
| `comparison_optimized.png` | Optimized input waveform (top) and acoustic response (bottom) |
| `optimized_params.npy` | Saved Fourier coefficients for best result |

---

## Observations

1. **Amplitude vs Width Trade-off**: The optimizer strongly favored amplitude over width given the weighting (1.0 vs 0.3). The resulting pulse is wider but much stronger.

2. **Piezo Resonance**: The piezoelectric transducer's resonant behavior at 40kHz means it responds best to sustained excitation. A longer input waveform allows the mechanical system to build up oscillation amplitude.

3. **Ringdown**: The characteristic asymmetric decay (fast attack, slow ringdown) is visible in both optimized and unoptimized responses, but the optimized version has higher peak amplitude during the driven phase.

4. **Run-to-run Variability**: Measurements show some variability between runs (~10%), likely due to:
   - Acoustic coupling variations
   - Temperature effects on sound speed
   - Timing jitter in recording synchronization

---

## Next Steps

- [ ] Re-run with higher `width_weight` to prioritize narrower pulses
- [ ] Try more Fourier coefficients (12-16) for finer envelope control
- [ ] Implement burst mode with precise triggering for better synchronization
- [ ] Test reproducibility across multiple runs
- [ ] Characterize temperature dependence

---

*Log created: May 26, 2026*
