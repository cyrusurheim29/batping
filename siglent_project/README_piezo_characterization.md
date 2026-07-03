# Piezo Transducer Characterization

## Overview

`piezo_characterization.py` measures the frequency response of the piezoelectric transducer system using a chirp (frequency sweep) method. This determines the usable bandwidth for PRBS waveform design.

## Method

1. Generate a linear chirp sweeping 20-60 kHz over 10 ms
2. Upload to Siglent SDG1062X AWG at 50 Hz repetition rate (20 ms period)
3. Record response via UltraMic at 384 kHz
4. Compute transfer function H(f) = FFT(received) / FFT(transmitted)
5. Find -3 dB and -6 dB bandwidth from magnitude response

## Hardware Setup

- **AWG**: Siglent SDG1062X via LAN (10.11.13.230:5025)
- **Microphone**: UltraMic @ 384 kHz sample rate
- **Geometry**: Piezo and mic side-by-side, 1 m to flat reflector (2 m round-trip)

## Results (2026-06-14)

- Center frequency: 40.97 kHz
- -3 dB bandwidth: 2.02 kHz
- -3 dB passband: 39.97 - 41.99 kHz
- Q factor: ~20

## Time-Domain Signal Interpretation

The recorded signal shows two peaks per 20 ms period:

| Peak | Amplitude | Description |
|------|-----------|-------------|
| Small (~0.31) | Direct path | Piezo → mic (side-by-side, ~0 ms propagation) |
| Large (~0.86) | Reflected path | Piezo → reflector → mic (+5.8 ms propagation) |

The reflection is **stronger** than the direct signal because:
- The piezo radiates primarily forward (toward the reflector)
- The mic, positioned beside the piezo, is off-axis from the main beam
- The reflector redirects energy back toward the piezo/mic pair

### Timing

Within each chirp period:
- **t = 5 ms**: Chirp reaches 40 kHz (piezo resonance)
- **t ≈ 5 ms**: Small peak (direct path, negligible delay)
- **t ≈ 10.8 ms**: Large peak (reflected path: 5 ms + 5.8 ms round-trip)

### Asynchronous Recording Note

**Important**: The recording is NOT synchronized to the AWG output. The SDG1062X runs continuously, and `pyaudio` recording starts at an arbitrary phase relative to the chirp. This means:

1. Absolute times in the raw recording (e.g., "peak at 2.5 ms") have no physical meaning
2. Only **relative timing** between peaks is meaningful (e.g., ~6 ms between direct and reflected)
3. The period of 20 ms correctly matches the 50 Hz repetition rate

For time-of-flight measurements requiring absolute timing, consider:
- Using the SDG's SYNC output to trigger the recording
- Using a digital delay generator (e.g., SRS DG645) for precise synchronization

## Output Files

- `piezo_characterization.png` - Frequency response and time-domain plots
- `piezo_characterization.npz` - Raw data (freqs, H_mag, H_phase, bandwidth values, received signal)
- `piezo_chirp_response.wav` - Raw recorded audio

## Dependencies

- numpy, scipy, matplotlib
- pyaudio (for UltraMic recording)
- Standard socket library (for SDG LAN control)
