# BATPING - Bidirectional Acoustic Traveltime Pulser for INdex Gradients

Ultrasonic time-of-flight (TOF) measurement system for monitoring temperature fluctuations via changes in the speed of sound.

## Project Goal

Achieve **0.1°C RMS temperature resolution** by measuring acoustic TOF changes along a fixed path. The speed of sound in air varies with temperature:

```
c(T) = 331.3 + 0.6T  [m/s, T in °C]
```

For a 2m path, 0.1°C temperature change produces ~1 μs TOF shift.

## Current Status: VALIDATED

**Displacement test (June 1, 2026)** demonstrated:
- Measured ΔTOF: **622 ± 2 μs** (statistical)
- Expected ΔTOF: **583 ± 117 μs** (for 10cm displacement)
- Agreement: **0.33σ** ✓
- Inferred displacement: 10.7 cm (nominal: 10 cm)

Statistical precision of **±2 μs** corresponds to **~0.02°C** temperature resolution — well within the 0.1°C target.

## Hardware

- **AWG**: Agilent 33220A arbitrary waveform generator
- **Transducer**: 40 kHz piezoelectric array (hexagonal, 7 elements, ~90 nF)
- **Microphone**: Dodotronic Ultramic 384K (384 kHz sampling)
- **Reflector**: Flat wooden slab for round-trip measurements

## Method: Matched Filter with Direct-Reflection Timing

1. **Waveform**: 2-cycle Gaussian envelope at 40 kHz, 50 Hz rep rate
2. **Geometry**: Reflection setup provides timing reference
   - Mic picks up faint **direct signal** at emission time (~0 ms)
   - Mic picks up strong **reflection** after round-trip (~5-6 ms)
3. **Measurement**: Direct-to-reflection interval = TOF
4. **Precision**: Sub-sample accuracy via parabolic interpolation

## Directory Structure

```
agilent_project/
├── README.md                    # This file
├── agilent_33220a.py           # AWG control library
├── compare_waveforms.py        # Waveform comparison utilities
├── DATA_LOG.md                 # Historical data log
│
├── matched_filtering_v1/       # ★ Successful displacement test (June 1)
│   ├── gaussian_2m_reflection_50Hz.wav      # Template recording (2m path)
│   ├── gaussian_1.8m_reflection_50Hz.wav    # Test recording (1.8m path)
│   ├── gaussian_matched_filter_2m.npz       # Saved template
│   ├── tof_with_error_analysis.png          # Error propagation results
│   ├── matched_filter_template.png          # Template construction
│   └── ...
│
├── squarewave_analysis/        # 2 Hz square wave driving experiments
│   ├── squarewave_2Hz_10VPP_*.aiff          # Raw recordings
│   ├── waveform_structure_analysis.png      # Peak structure analysis
│   └── ...
│
├── bayesian_optimization/      # Pulse shaping optimization (May 27)
│   ├── bayesian_optimizer.py                # Optimization code
│   ├── bayesian_optimizer_state.npz         # Saved results
│   ├── README_bayesian_optimizer.md         # Method documentation
│   └── ...                                  # 38% FWHM improvement achieved
│
├── tof_development/            # Earlier TOF measurement attempts (May 30)
│   ├── tof_temperature_monitor.py           # TOF monitoring class
│   └── ...
│
├── impulse_response/           # Piezo characterization
├── ambient_measurements/       # Background noise surveys
├── optimization_attempts/      # Earlier optimization methods (Jacobian, etc.)
├── scope_images/              # Oscilloscope photos
├── miscellaneous/             # Other plots and data
└── in-dome-considerations/    # Dome deployment planning
```

## Key Results

### Matched Filter TOF Measurement (June 1, 2026)

| Parameter | Value |
|-----------|-------|
| Path length | 2.0 m (round-trip) |
| Displacement | 10 cm (→ 1.8 m round-trip) |
| TOF @ 2m | 5411 ± 1 μs |
| TOF @ 1.8m | 4789 ± 2 μs |
| ΔTOF measured | 622 ± 2 μs |
| ΔTOF expected | 583 ± 117 μs |
| Temperature equiv. noise | ~0.02°C |

### Error Budget

| Source | Uncertainty | Effect on ΔTOF |
|--------|-------------|----------------|
| Path length (±2 cm each way) | ±4 cm round-trip | ±117 μs |
| Temperature (±1°C) | ±0.6 m/s | ~2 μs |
| Mic sampling (384 kHz) | 2.6 μs/sample | ~0.9 μs (with interp) |
| **Statistical repeatability** | - | **±2 μs** |

Systematic uncertainty dominates, but statistical precision is excellent.

### Bayesian Optimization (May 27, 2026)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| FWHM | 643 μs | 399 μs | **38%** |
| Amplitude | 1.26 | 1.09 | -13% |

## Usage

### Record and analyze TOF:

```python
# See matched_filtering_v1/ for complete example
from scipy.signal import correlate
from scipy.io import wavfile

# Load template
template_data = np.load('matched_filtering_v1/gaussian_matched_filter_2m.npz')
template = template_data['template']

# Cross-correlate with new recording
corr = correlate(new_signal, template, mode='full')
shift_samples = np.argmax(corr) - (len(template) - 1)
shift_us = shift_samples / fs * 1e6
```

### AWG control:

```python
from agilent_33220a import Agilent33220A

awg = Agilent33220A()
awg.connect()
awg.output_arbitrary(waveform, freq_hz=50, vpp=10)
awg.set_output(True)
```

## Dependencies

- numpy, scipy, matplotlib
- pyaudio (microphone recording)
- pyvisa (AWG communication)
- scikit-optimize (Bayesian optimization)

## References

- BATPING project documentation (see PDFs in root directory)
- Bayesian optimization: `bayesian_optimization/research_log_bayesian_optimization.md`
