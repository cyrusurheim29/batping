# BATPING In-Dome Considerations - Research Log

This document archives research findings, conversation excerpts, and analysis related to deploying the BATPING system in the telescope dome environment.

---

## Historical Context

### Project Overview

BATPING is an acoustic temperature measurement system for telescope dome seeing prediction using ultrasonic time-of-flight measurements. The speed of sound depends on air temperature, so by precisely measuring how long an ultrasonic pulse takes to travel a known distance, we can infer the air temperature.

For accurate time-of-flight measurements, we need pulses that are:
1. High amplitude (strong signal, good SNR)
2. Narrow width (precise timing)

### Hardware Components

- **Waveform Generator**: Agilent 33220A 20MHz (current), previously WaveStation 2052
- **Microphone**: Dodotronic Ultramic 384K EVO (384kHz sampling)
- **Transducer**: 40kHz piezoelectric array (resonant frequency)

### Earlier Work: 33kHz Motion Sensor Interference (March 2026)

From `batping/WORKLOG.md`:

> #### Environmental Interference
> - **33kHz interference** identified in Physics Building (motion sensor lights)
> - **Clean environment** found at home (hallway) - no 33kHz interference
>
> #### Key Measurements
>
> **Physics Building (Mar 17)**
> - 40 kHz signal: **-3.3 dB**
> - 33 kHz interference: **-11.8 dB**
> - SNR: ~8.5 dB
> - Beat frequency: ~7 kHz (period ≈ 0.143 ms)
>
> **Hallway Mirror (Mar 22) - Clean Environment**
> - 40 kHz signal: **-8.5 dB**
> - 33 kHz interference: **-75.1 dB** (noise floor)
> - SNR: **~66.6 dB**

### Time-Gating Methodology

Time gating isolates the acoustic signal based on expected arrival time, rejecting:
- EM coupling spike at t≈0 (speed of light)
- Continuous interference (motion sensors at 33 kHz)
- Later reverberations

From earlier documentation:
```
Time gate: [28ms -------- 45ms]
                  ^^^^^^^
                  Only analyze this region
```

This works for **out-of-band** interference (33kHz) because:
1. A bandpass filter (38-42kHz) rejects it before analysis
2. Time gating then isolates the pulse arrival window
3. The 33kHz energy is already gone by the time we analyze

---

## Session: May 26, 2026 - 40kHz HVAC Interference Discovery

### Setup

Working with relocated equipment at a new building:
- Transducer array and mic pointed at each other, ~22cm apart
- Agilent 33220A connected via USB
- Ultramic 384K recording at 384kHz

### Initial Observation: Persistent 40kHz Background

During pulse testing, observed that the 40kHz bandpass-filtered signal never dropped to zero between pulses. Investigation began.

#### Test 1: AWG Output OFF

> **Claude**: Let me test by turning off the AWG output completely and recording just ambient.
>
> **Results**:
> ```
> 40kHz band with AWG OFF:
>   RMS: 0.0862
>   Peak envelope: 0.1380
>   Mean envelope: 0.1218
>
> Top ultrasonic frequencies (AWG OFF):
>   40.0 kHz: 1162.2
> ```
>
> There's a strong **ambient 40kHz source** in your environment — with the AWG completely off, there's still a massive spike right at 40kHz.

#### Test 2: Transducer Disconnected

> **User**: could it be the whir from the agilent? you're right there's motion sensor in the room but i thought we classified that as 33kHz noise. the other thing is (recall our conversations + readme files to refresh on this) the sonic transducer array has a resonant frequency of 40kHz.
>
> **Claude**: The piezo transducer array is a resonator at 40kHz — it could be acting as both a source and a "tuning fork" that picks up ambient acoustic/mechanical energy and re-emits at its resonant frequency.
>
> **User**: transducer array disconnected
>
> **Results**:
> ```
> 40kHz band (transducer disconnected):
>   RMS: 0.1316
>   40kHz magnitude: 1774.5
>
> Comparison (transducer connected vs disconnected):
>   40kHz RMS: 0.0862 -> 0.1316
>   40kHz mag: 1162.2 -> 1774.5
> ```
>
> The 40kHz is still there and actually *stronger* with the transducer disconnected (1774 vs 1162). So it's not the transducer resonating.

#### Test 3: Agilent Unplugged

> **User**: [unplugged Agilent from wall]
>
> **Results**:
> ```
> Results (Agilent unplugged):
>   40kHz band RMS: 0.0219
>   33kHz magnitude: 0.8
>   40kHz magnitude: 234.9
>   80kHz magnitude: 8.0
>
> Comparison:
>   Transducer connected, AWG on/off: 40kHz mag = 1162
>   Transducer disconnected, AWG off: 40kHz mag = 1774
>   Agilent unplugged:                40kHz mag = 235
> ```
>
> **Claude**: The Agilent is the culprit. The 40kHz dropped by ~7x when the Agilent was unplugged. The function generator's internal switching power supply or clock circuitry is emitting 40kHz.

#### Test 4: Mic Moved Away from Computer

> **User**: im not so sure that unplugging the agilent completely fixed the issue. my interpretation of ambient_no_agilent.png is that the 40kHz peak is still quite strong (on order of 10^2), which is even more significant than the 33kHz motion sensor peak. additionally we're still seeing a peak at the first harmonic as well (80kHz). im wondering whether the peak might be environmental, or from, say, my computer.
>
> **Results (mic moved)**:
> ```
> Results (mic moved away):
>   33kHz magnitude: 0.4
>   40kHz magnitude: 224.3
>   80kHz magnitude: 0.9
>
> Comparison to previous:
>   Agilent unplugged (mic at table): 40kHz = 235
>   Mic moved away:                   40kHz = 224
> ```
>
> The 40kHz is still at ~224 — essentially unchanged. So it's not the computer either.

### Multi-Location Survey

> **User**: okay, after doing some noise characterization in different rooms (including moving from the third floor of the building i am in to its second floor), i think i've concluded that the noise was coming from some turbulent airflow or leakage from the hvac system, which has its hub on the fourth floor.

#### Location Comparison Results

| Location | 33 kHz | 40 kHz |
|----------|--------|--------|
| Third Floor Office (near HVAC) | -58.6 dB | **-28.3 dB** |
| Second Floor Hallway | **-15.9 dB** | -75.3 dB |
| Second Floor Atrium | **-9.8 dB** | -54.3 dB |
| Second Floor Office | **-11.7 dB** | -71.3 dB |

**Key finding**: Third floor dominated by 40kHz HVAC noise (-28 dB), while second floor has 40kHz drop of ~45 dB but louder 33kHz motion sensors.

See: `plots/location_comparison.png`

---

## Analysis: Why Time-Gating Doesn't Work for 40kHz Interference

> **User**: i have a question: why weren't we able to also time gate out the 40kHz noise? my understanding for why it was trickier than the 33kHz noise is that it's right at the resonant frequency of the piezo transducer array, which would be problematic if we ever want to drive a waveform with some 40kHz feature (which im assuming we will indeed want to do...).

**Claude's analysis**:

Time gating works for the 33kHz because:
1. It's at a **different frequency** than our signal
2. It's **continuous** — present throughout the recording
3. The bandpass filter (38-42kHz) rejects it before we even apply the time gate

The 40kHz HVAC noise is problematic because:
1. It's **at our carrier frequency** — the bandpass filter keeps it
2. Time gating isolates a *time window*, not a frequency — the 40kHz noise is present in that window too
3. Our pulsed signal adds *on top* of the continuous 40kHz interference

---

## Proposed Strategies for Dome Deployment

> **User**: in the future implementation of this project, say in the telescope dome for example, how would we get around similar issues? it's not like we can just relocate the setup as i just did.

### 1. Background Subtraction
Record a reference with no pulse, subtract it from the pulsed recording. This cancels continuous interference regardless of frequency.

### 2. Lock-in Detection / Synchronous Averaging
Trigger recording precisely with pulse transmission, average many pulses. Random noise averages to zero; our coherent signal adds constructively.

### 3. Frequency Hopping
Shift carrier slightly (e.g., 39kHz or 41kHz) if interference is narrowband. The piezo has bandwidth — it won't be as efficient but will still work.

### 4. Environment Characterization Before Deployment
Survey the ultrasonic spectrum at the dome before finalizing the design. Telescope domes may actually be quieter in the 40kHz band (no HVAC in the same way).

### 5. Burst Mode with Gating
Use triggered bursts and only analyze the time window where we *expect* our signal. Combined with background subtraction, this is quite robust.

---

## Plots Index

All plots stored in `plots/` subdirectory:

| File | Description |
|------|-------------|
| `location_comparison.png` | Multi-location spectrum comparison (key summary plot) |
| `ambient_ultrasonic.png` | Initial ambient spectrum showing 40kHz with AWG off |
| `ambient_no_agilent.png` | Spectrum with Agilent fully unplugged |
| `ambient_no_transducer.png` | Spectrum with transducer disconnected |
| `ambient_mic_moved.png` | Spectrum with mic repositioned away from computer |
| `may26_thirdflooroffice.txt` | Spectrum data: 3rd floor office (high 40kHz) |
| `may26_secondfloorhallway.txt` | Spectrum data: 2nd floor hallway |
| `may26_secondflooratrium.txt` | Spectrum data: 2nd floor atrium |
| `may26_secondflooroffice.txt` | Spectrum data: 2nd floor office |

---

## Analysis: Pulse Width Metrics for Ringdown Signals (May 26, 2026)

### The Problem

The piezoelectric transducer exhibits characteristic **ringdown** behavior: a sharp initial peak followed by a slowly decaying tail. The standard FWHM measurement captures the entire duration where the envelope exceeds half-maximum, which may not reflect the **precision of arrival time detection**.

Current FWHM algorithm:
1. Find global peak of envelope
2. Calculate `half_max = peak_amplitude / 2`
3. Find last rising crossing before peak
4. Find first falling crossing after peak
5. FWHM = time between crossings

For ringdown signals where the tail stays above half_max:
```
     Peak
      /\
     /  \______________  (ringdown tail above half_max)
    /                  \_____
```
This gives an inflated FWHM that doesn't reflect the sharpness of the initial attack.

### Alternative Approaches (for PI discussion)

1. **Peak-to-first-zero-crossing**: Measure width of just the initial peak (attack + first half of decay). Ignores tail.

2. **10-90% rise time**: How fast the signal rises from 10% to 90% of peak. Directly measures attack sharpness, completely ignores tail.

3. **Threshold-based arrival detection**: Time when signal first crosses a fixed threshold. This is what actually matters for TOF precision — how cleanly can we identify the arrival instant?

4. **Separate metrics (current choice)**: Report both:
   - "Attack width": Width of initial peak only
   - "Total duration": Full FWHM including ringdown
   
   This preserves information and doesn't prematurely commit to a single metric.

### Relevance to Optimization

The optimizer's objective function uses FWHM in the penalty term. If FWHM captures the ringdown tail, the optimizer may not effectively minimize the **attack width** that matters for TOF precision. This could explain why the optimizer converged to a near-continuous waveform — it maximizes amplitude without being penalized for the long ringdown, since the ringdown duration is similar regardless of input shape.

**Action item**: Implement separate attack/ringdown metrics and re-evaluate optimization objective function.

---

## Future Work

*This section will be updated as hardware concerns and implementation ideas develop over Summer 2026.*

- [ ] Characterize ultrasonic environment at telescope dome
- [ ] Implement background subtraction in pulse_optimizer.py
- [ ] Test synchronous averaging with triggered bursts
- [ ] Evaluate alternative carrier frequencies (38kHz, 42kHz)
- [ ] Hardware shielding considerations
- [ ] Implement separate attack width / total duration metrics
- [ ] Discuss pulse width metric alternatives with PI

---

*Last updated: May 26, 2026*
