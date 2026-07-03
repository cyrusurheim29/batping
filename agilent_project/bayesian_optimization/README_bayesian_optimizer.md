# Bayesian Optimization for BATPING Pulse Shaping

## Overview

This directory contains tools for optimizing the acoustic pulse shape emitted by the BATPING piezoelectric transducer array. The goal is to minimize the full-width at half-maximum (FWHM) of the acoustic pulse envelope to improve time-of-flight measurement precision.

## Background

The piezoelectric transducers exhibit resonant behavior at 40 kHz, causing significant ringdown that broadens the acoustic pulse. A naive Gaussian input pulse produces an output with FWHM ~600-700 μs, limited by the transducer's Q-factor.

Previous attempts using Jacobian-based gradient descent (iterative local tangent estimation) failed because:
1. The L2 shape-matching objective allowed the optimizer to "game" the metric
2. The target Gaussian was physically unachievable given piezo constraints
3. Each iteration required ~270 measurements for Jacobian estimation (~5 min/iteration)

## Bayesian Optimization Approach

We use Gaussian Process (GP) regression to build a surrogate model of the objective function, then optimize using Expected Improvement (EI) acquisition.

### Key Features

- **Direct FWHM objective**: Optimizes what we actually care about
- **Low-dimensional parameterization**: 10 parameters instead of 2700 samples
- **Sample-efficient**: ~100 evaluations total vs. 270 per Jacobian iteration
- **Automatic exploration/exploitation**: GP uncertainty guides search

### Parameterization

The input waveform is parameterized by:
- `sigma`: Gaussian envelope width (1-15 cycles)
- `phase`: Carrier phase (0-2π)
- `pre_comp[3]`: Pre-pulse compensation amplitudes
- `post_comp[5]`: Post-pulse ringdown cancellation amplitudes

## Results (2026-05-27)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| FWHM | 643 μs | 399 μs | **38%** |
| Amplitude | 1.26 | 1.09 | -13% (within tolerance) |

The optimizer discovered that:
- Narrow input pulses (σ ≈ 1.2 cycles) work best
- Aggressive post-pulse compensation (saturating at -0.5) helps cancel ringdown
- Pre-pulse compensation provides minor additional sharpening

## Files

- `bayesian_optimizer.py` - Main optimization script
- `bayesian_optimizer_state.npz` - Saved optimization results
- `waveform_comparison_data.npz` - Baseline vs optimized measurements
- `bayesian_optimization_results.png` - Convergence and summary plot
- `waveform_comparison_overlay.png` - Side-by-side input/output comparison
- `baseline_waveform_detail.png` - Baseline input/output detail
- `best_waveform_detail.png` - Optimized input/output detail

## Usage

```bash
# Run optimization (takes ~20 min for 100 evaluations)
python bayesian_optimizer.py

# Results saved to bayesian_optimizer_state.npz
```

## Dependencies

- numpy, scipy, matplotlib
- scikit-optimize (`pip install scikit-optimize`)
- pyaudio (for microphone recording)
- pyvisa (for AWG communication)

## Hardware

- Agilent 33220A arbitrary waveform generator
- Dodotronic Ultramic 384K microphone
- 40 kHz piezoelectric transducer array (hexagonal configuration)
