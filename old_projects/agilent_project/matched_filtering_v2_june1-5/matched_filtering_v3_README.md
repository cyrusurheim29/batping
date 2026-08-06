# Matched Filter TOF Measurement - v1 (June 1, 2026)

Successful demonstration of displacement detection via matched filter cross-correlation.

## Experiment Setup

- **Geometry**: Reflection-based round-trip measurement
- **Initial path**: 2.0 m round-trip (1.0 m to reflector)
- **Displacement**: Reflector moved 10 cm closer → 1.8 m round-trip
- **Waveform**: 2-cycle Gaussian envelope at 40 kHz
- **Rep rate**: 50 Hz (20 ms period)
- **Amplitude**: 10 Vpp

## Method

1. **Template creation**: Record 10 pulses at 2m, average envelopes aligned to direct pickup
2. **Test recording**: Move reflector 10 cm closer, record new pulses
3. **Cross-correlation**: Align signals to DIRECT pickup (not reflection), cross-correlate
4. **Peak detection**: Sub-sample precision via parabolic interpolation

**Critical insight**: Must align to direct pickup, not reflection peak. Aligning to reflection cancels out the shift we're trying to measure.

## Results

### Measured Values

| Recording | TOF (μs) | σ (μs) | n |
|-----------|----------|--------|---|
| 2.0 m path | 5411 | 1.0 | 15 |
| 1.8 m path | 4789 | 1.6 | 15 |
| **ΔTOF** | **622** | **1.9** | - |

### Comparison with Expected

| Parameter | Expected | Measured |
|-----------|----------|----------|
| ΔTOF | 583 ± 117 μs | 622 ± 2 μs |
| Displacement | 10.0 cm | 10.7 cm (inferred) |
| Agreement | - | **0.33σ** ✓ |

## Error Analysis

### Systematic Uncertainties

| Source | Value | Uncertainty | Effect on ΔTOF |
|--------|-------|-------------|----------------|
| Path length | 2.0 m | ±4 cm | ±117 μs |
| Displacement | 0.2 m | ±4 cm | ±117 μs |
| Temperature | 23°C | ±1.1°C | ~2 μs |
| Speed of sound | 343 m/s | ±0.7 m/s | ~2 μs |

### Statistical Uncertainties

| Source | Value |
|--------|-------|
| Mic sampling | 2.60 μs/sample |
| With interpolation | ~0.87 μs |
| AWG clock (100 ppm) | ~0.54 μs |
| **Measured repeatability** | **±1.9 μs** |

### Combined Uncertainty

- Systematic: ±117 μs (dominates)
- Statistical: ±1.9 μs
- Total: ±117 μs

## Temperature Equivalent

For a 2 m path:
- ΔT = Δτ × c² / (0.6 × d)
- Statistical precision of ±2 μs → **~0.02°C** noise floor
- Target was 0.1°C → **5× margin achieved**

## Files

| File | Description |
|------|-------------|
| `gaussian_2m_reflection_50Hz.wav` | Template recording (2m path) |
| `gaussian_1.8m_reflection_50Hz.wav` | Test recording (1.8m path) |
| `gaussian_matched_filter_2m.npz` | Saved template (envelope) |
| `tof_with_error_analysis.png` | Main results figure |
| `matched_filter_template.png` | Template construction |
| `gaussian_50Hz_analysis.png` | Full recording overview |
| `gaussian_50Hz_single_period.png` | Single period detail |
| `intended_waveform.png` | AWG output waveform |
| `check_1.8m_recording.png` | 1.8m recording verification |

## Reproduction

```python
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, find_peaks, correlate

def bandpass_envelope(signal, fs, low=35000, high=45000):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return envelope

# Load recordings
fs1, data1 = wavfile.read('gaussian_2m_reflection_50Hz.wav')
fs2, data2 = wavfile.read('gaussian_1.8m_reflection_50Hz.wav')

# Compute envelopes
env1 = bandpass_envelope(data1.astype(float)/32768, fs1)
env2 = bandpass_envelope(data2.astype(float)/32768, fs2)

# Find direct-reflection pairs and measure intervals
# (See main analysis code for full implementation)
```

## Lessons Learned

1. **Align to emission, not arrival**: Cross-correlation must align signals at the direct pickup, not the reflection peak
2. **Reflection setup works**: Provides self-referencing timing without knowing AWG phase
3. **Sub-sample interpolation essential**: Parabolic fit reduces quantization error from 2.6 μs to <1 μs
4. **Systematic errors dominate**: Path length uncertainty (±2 cm) >> statistical precision (±2 μs)
