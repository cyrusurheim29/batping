# Batping Project Work Log

## Session: March 27, 2026

### Project Overview
Acoustic temperature measurement system for telescope dome seeing prediction using ultrasonic time-of-flight measurements.

---

### Hardware Setup
- **Waveform Generator**: Agilent 20MHz
- **Microphone**: Dodotronic Ultramic 384K EVO (384kHz sampling)
- **Transducer**: 40kHz piezoelectric array
- **Reflector**: Whiteboard (for bistatic tests), corner reflector (planned for monostatic)

### Key Issues Resolved

#### 1. Microphone Saturation
- **Problem**: Ultramic showing saturated signal even in quiet room
- **Cause**: Gain set too high + ultrasonic interference from environment
- **Solution**: Manually reduced gain to lowest setting on Dodotronic mic

#### 2. Sampling Rate
- **Problem**: Could not see 40kHz signal at 44.1kHz sample rate
- **Cause**: Nyquist frequency (22.05kHz) below signal frequency
- **Solution**: Set Audacity/Audio MIDI Setup to 384kHz sample rate

#### 3. USB Adapter Issues
- **Problem**: Potential noise from cheap USB-A to USB-C adapters
- **Notes**: Tried Staples adapter, then Ainope + Belkin combo
- **Result**: Still had 33kHz interference, but workable after gain adjustment

#### 4. Environmental Interference
- **33kHz interference** identified in Physics Building (motion sensor lights)
- **Clean environment** found at home (hallway) - no 33kHz interference

---

### Data Files

| File | Description |
|------|-------------|
| `cu_data_mar19_40khz_520mVp2p_audio.aiff` | Initial test, 40kHz @ 520mV p2p drive |
| `cu_data_mar19_40khz_520mVp2p_spectrum.txt` | Spectrum export from Audacity |
| `iu_lab_aligned_mar16.aiff` | Physics Building recording (direct path) |
| `spectrum_iu_lab_mar17.txt` | Physics Building spectrum data |
| `mar22_hallway_mirror.aiff` | Home hallway, reflected off whiteboard |
| `mar22_hallway_mirror.txt` | Hallway mirror spectrum data |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze_signal.py` | Full signal analysis with auto-detection of signal region, bandpass filtering |
| `quick_spectrum.py` | Quick spectrum plot from AIFF files |
| `plot_spectra.py` | Audacity-style filled spectrum plots from txt files |
| `plot_beats.py` | Waveform plot showing beat pattern |

### Generated Plots

| Plot | Description |
|------|-------------|
| `cu_data_mar19_40khz_520mVp2p_audio_analysis.png` | 4-panel analysis of initial test |
| `spectrum_iu_lab_mar17.png` | Power spectrum - Physics Building |
| `spectrum_hallway_mirror_mar22.png` | Power spectrum - Hallway (clean) |
| `iu_lab_beats_waveform.png` | Waveform showing 7kHz beats |

---

### Key Measurements

#### Physics Building (Mar 17)
- 40 kHz signal: **-3.3 dB**
- 33 kHz interference: **-11.8 dB**
- SNR: ~8.5 dB
- Beat frequency: ~7 kHz (period ≈ 0.143 ms)

#### Hallway Mirror (Mar 22) - Clean Environment
- 40 kHz signal: **-8.5 dB**
- 33 kHz interference: **-75.1 dB** (noise floor)
- SNR: **~66.6 dB**

---

### Terminology
- **Direct path**: Mic positioned directly in front of piezo speaker
- **Reflected/Bistatic**: Mic at L-angle, reading signal off reflector
- **Monostatic**: TX and RX co-located (planned setup with corner reflector)

### Next Steps
1. Increase drive voltage (currently 520mV p2p, target ~10V p2p via transformer)
2. Set up corner reflector for monostatic echo detection
3. Implement pulse/burst mode for time-of-flight measurements
4. Characterize ultrasonic environment at telescope dome

---

### Technical Notes

#### Why dB levels are negative (dBFS)
- 0 dBFS = maximum digital amplitude (clipping)
- Negative values = below full scale
- -3 dB ≈ 71% of max, -6 dB ≈ 50%, -20 dB ≈ 10%

#### Beat Frequency
When 40kHz signal mixes with 33kHz interference:
- Beat frequency = |40 - 33| = 7 kHz
- Creates amplitude modulation visible in time-domain waveform

---

### References
- PI's work log (May 2024): Initial proof of concept with MATLAB analysis
- Herchel Smith Research Proposal (in folder)
