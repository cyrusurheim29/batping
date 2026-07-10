#!/usr/bin/env python3
"""Build BATPING presentation as .pptx"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import os

BASE = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project'
AGILENT = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project'

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]   # completely blank
TITLE_ONLY = prs.slide_layouts[5]

# Colors
NAVY  = RGBColor(0x1a,0x1a,0x4e)
RED   = RGBColor(0xc0,0x39,0x2b)
GRAY  = RGBColor(0x55,0x55,0x55)
WHITE = RGBColor(0xff,0xff,0xff)
LTGRAY= RGBColor(0xf2,0xf2,0xf2)

def add_slide():
    return prs.slides.add_slide(BLANK)

def title_box(slide, text, y=Inches(0.3), fontsize=32, color=NAVY):
    txb = slide.shapes.add_textbox(Inches(0.5), y, Inches(12.3), Inches(0.7))
    tf = txb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]; p.text = text
    p.font.bold = True; p.font.size = Pt(fontsize); p.font.color.rgb = color
    return txb

def body_box(slide, text, left=Inches(0.5), top=Inches(1.2),
             width=Inches(12.3), height=Inches(5.8), fontsize=18, color=GRAY):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame; tf.word_wrap = True
    # Split on newlines — first line starts paragraph
    lines = text.strip().split('\n')
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(fontsize); p.font.color.rgb = color
        if line.startswith('•') or line.startswith('–'):
            p.level = 1
    return txb

def add_image(slide, path, left, top, width=None, height=None):
    if not os.path.exists(path):
        print(f"  MISSING: {path}"); return
    if width and height:
        slide.shapes.add_picture(path, left, top, width, height)
    elif width:
        slide.shapes.add_picture(path, left, top, width=width)
    elif height:
        slide.shapes.add_picture(path, left, top, height=height)
    else:
        slide.shapes.add_picture(path, left, top)

def rule(slide, y=Inches(1.05), color=NAVY):
    line = slide.shapes.add_shape(1, Inches(0.5), y, Inches(12.3), Pt(2))
    line.line.color.rgb = color; line.fill.background()

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
txb = sl.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.5))
tf = txb.text_frame
p = tf.paragraphs[0]; p.text = "BATPING"
p.font.size = Pt(52); p.font.bold = True; p.font.color.rgb = NAVY; p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "Bidirectional Acoustics Traveltime Pulser for INdex Gradients"
p2.font.size = Pt(22); p2.font.color.rgb = GRAY; p2.alignment = PP_ALIGN.CENTER
p3 = tf.add_paragraph(); p3.text = ""
p4 = tf.add_paragraph()
p4.text = "Cyrus Urheim  ·  Stubbs Group  ·  Harvard  ·  July 2026"
p4.font.size = Pt(16); p4.font.color.rgb = GRAY; p4.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2: Introduction
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Introduction")
rule(sl)
body_box(sl, """Statement of Purpose
• LSST/Rubin dome and mirror seeing remains uncharacterized (Urbach et al. 2026)
• Temperature fluctuations → refractive index gradients → PSF broadening
• BATPING measures the line integral of temperature along the optical path via acoustic TOF

Precision Goal
• Target: σ_T < 0.1 °C per ping
• Equivalent to σ_TOF < 1.02 μs  (for 2m round-trip path at 20 °C)
• Best achieved so far: σ(Δτ) = 0.076 μs → 0.007 °C/ping  (Stubbs pipeline, a0.8_sm_2ms)

Anticipated Final Implementation
• Winfield Hill transponder PCB: retransmits received burst after known delay → optimal adaptive matched filter
• Array of piezo transducers (hexagonal, ~40 kHz) + Dodotronic UltraMic 384k
• Rubin dome crossing paths; continuous monitoring during observations""",
fontsize=16)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3: Progress Checklist (blank — Cyrus to fill)
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Progress Since Last Meeting")
rule(sl)
body_box(sl, """[Cyrus to fill in]

□
□
□
□
□
□ """, fontsize=18)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4: Waveform Design Rationale — Early Approaches
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Waveform Design Rationale (I): Early Approaches")
rule(sl)

body_box(sl, """Modified Gaussian + Bayesian Optimization (v1–v3)
• Parametrized waveform: Gaussian envelope × 40 kHz carrier + K sinusoidal compensation bursts
• Objective: minimize FWHM of ACF peak  (then per-period σ_TOF)
• v3 (N=1000 iter): 38% FWHM improvement  (443 → 275 μs) but marginal gains beyond v2 — parametrization bottleneck

Barker-7 Code (BPSK modulated onto 40 kHz carrier)
• Sidelobe level: −12 to −14 dB (theoretical −16.9 dB)
• FWHM ∝ T_chip = N_cyc / f_carrier — scales linearly, no improvement over Gaussian at matched chip size
• Both approaches hit a ~250 μs FWHM floor set by the piezo ring-down (τ_ring ≈ Q/πf₀ ≈ 180 μs)""",
left=Inches(0.5), top=Inches(1.2), width=Inches(6.2), height=Inches(5.5), fontsize=15)

# Gaussian plot
add_image(sl,
    f'{AGILENT}/bayesian_optimization/waveform_comparison_overlay.png',
    Inches(7.0), Inches(1.2), width=Inches(5.8))

# Barker plot (small, below)
add_image(sl,
    f'{BASE}/barker7_32cyc_v5.png',
    Inches(7.0), Inches(4.2), width=Inches(5.8), height=Inches(2.8))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5: Waveform Design Rationale — LFM Chirp (cont.)
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Waveform Design Rationale (II): LFM Chirped Waveforms")
rule(sl)

body_box(sl, r"""Why a chirp?
• Sweeps f_start → f_stop over duration T: band-limited approximation to a delta function's spectrum
• Prof. Stubbs: "a delta function spread out in time"

Expected autocorrelation peak shape (Wiener-Khinchin):
  |S(f)|² ≈ rect((f − f₀)/B)   →   y(τ) = IFT(|S(f)|²) = B · sinc(Bτ)

Expected FWHM:
  sinc²(Bτ) = 0.5  →  Bτ ≈ 0.443  →  FWHM = 0.886 / B

Amplitude compensation:
• Flat: constant amplitude — simple, surprisingly competitive
• comp (1/H(f), 10% floor): flattens acoustic output spectrum
• comp_light (20% floor): less aggressive, less ring-down
• α-parametrization: H(f)^{−α},  α ∈ [0,1] — continuous flat→full-inversion

Key caveat:
  σ_TOF ≈ FWHM / (3.33 × SNR)
  Aggressive compensation amplifies off-resonance noise as much as signal
  → wider σ_TOF despite narrower FWHM if SNR degrades""",
left=Inches(0.5), top=Inches(1.2), width=Inches(7.5), height=Inches(5.5), fontsize=14)

# Spectrum figure
add_image(sl,
    f'{BASE}/spectrum_flat_chirp.png',
    Inches(8.2), Inches(1.2), width=Inches(4.8))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6: Piezo Transfer Function
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Piezoelectric Transducer: Frequency Response H(f)")
rule(sl)
add_image(sl, f'{BASE}/piezo_characterization.png',
          Inches(1.5), Inches(1.1), height=Inches(5.9))

# Small caption
txb = sl.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3))
tf = txb.text_frame; p = tf.paragraphs[0]
p.text = "Stepped-sine characterization (20–65 kHz, 50 Hz steps) · f₀ = 41.3 kHz · Q ≈ 22.8 · −3 dB BW = 2.95 kHz · two-way peak group delay ≈ 343 μs"
p.font.size = Pt(10); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7: Metrics Definitions
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Measurement Metrics")
rule(sl)
body_box(sl, r"""FWHM_corr   Full width at half max of TOF peak in per-period autocorrelation.
              11-pt least-squares parabolic fit (DoF = 8) around argmax gives sub-sample
              precision. Single-ping measurement; independent of between-ping conditions.

δ_peak      1σ uncertainty on FWHM_corr from the parabola fit residuals:
              δ_peak = RMS_residual / (|a| · √Σx²)   where a is the quadratic coefficient.
              Typical values: 15–300 ns.

σ_TOF       Standard deviation of per-period TOF estimates across N consecutive pings.
              The key temperature precision metric:
              σ_T = σ_TOF / (2L · (dc/dT) / c²) ≈ σ_TOF / 10.3 μs·°C⁻¹

σ(Δτ)       Stubbs pipeline: σ(τ_return − τ_launch) over evaluation periods.
              Cancels common-mode alignment jitter. Better than σ_TOF for long recordings.

SNR (CR)    Cramér-Rao implied SNR: FWHM / (3.33 × σ_TOF).
              Measures how close σ_TOF is to the instrument noise floor.
              meas/CR ≈ 1–3: instrument-limited.  meas/CR >> 3: environment-limited.""",
fontsize=15)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8: Methods Overview — Autocorr & Stubbs pipelines
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Analysis Methods")
rule(sl)

body_box(sl,
"""Per-period autocorrelation (fast, online)
• Autocorrelate each received period with itself
• TOF peak at lag ≈ 5.7 ms (2m round-trip); FWHM and position extracted with 11-pt parabolic fit
• σ_TOF from scatter of per-period peak positions

Stubbs matched-filter pipeline (30s, offline)
• Adaptive period slicer: finds actual incident peaks, not fixed-boundary clocks (fixes SDG↔UltraMic drift)
• Co-add 100 aligned periods → T_launch, T_return templates (√N SNR gain)
• Evaluate: xcorr each period with templates → τ_launch, τ_return, Δτ = τ_R − τ_L
• τ_launch ≈ 0 is alignment diagnostic; σ(Δτ) = temperature precision""",
left=Inches(0.5), top=Inches(1.2), width=Inches(5.8), height=Inches(3.2), fontsize=14)

# ppp_xcorr diagnostic — alignment failure modes
add_image(sl, f'{BASE}/ppp_xcorr_diagnostic.png',
          Inches(6.5), Inches(1.2), width=Inches(6.6))

txb = sl.shapes.add_textbox(Inches(6.5), Inches(4.4), Inches(6.6), Inches(0.4))
tf = txb.text_frame; p = tf.paragraphs[0]
p.text = "Three alignment failure modes in the Stubbs pipeline — all fixed in current implementation"
p.font.size = Pt(11); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

# period_comparison figure
add_image(sl, f'{BASE}/period_comparison_all3.png',
          Inches(0.5), Inches(4.6), width=Inches(5.8))

txb = sl.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(5.8), Inches(0.35))
tf = txb.text_frame; p = tf.paragraphs[0]
p.text = "Fixed: correct incident / reflected window identification across all waveform types"
p.font.size = Pt(11); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9: Best waveform — recording plot (bw_25kHz_comp)
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Results: bw_25kHz_wide_2ms_comp — Autocorrelation")
rule(sl)

txb = sl.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(5.0), Inches(0.5))
tf = txb.text_frame; p = tf.paragraphs[0]
p.text = "f = 29–54 kHz  |  BW = 25 kHz  |  T = 2 ms  |  TBP = 50  |  comp (1/H, 10% floor)  |  6.22 Vpp"
p.font.size = Pt(12); p.font.color.rgb = GRAY

add_image(sl,
    f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/bw_25kHz_wide_2ms_comp.png',
    Inches(0.4), Inches(1.5), height=Inches(5.8))

add_image(sl,
    f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/info_bw_25kHz_wide_2ms_comp.png',
    Inches(9.8), Inches(1.5), height=Inches(5.8))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10: Best waveform — Stubbs plot
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Results: bw_25kHz_wide_2ms_comp — Stubbs Matched Filter")
rule(sl)
add_image(sl,
    f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/stubbs_bw_25kHz_wide_2ms_comp_received.png',
    Inches(0.5), Inches(1.1), height=Inches(6.1))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11: a0.8_sm_2ms — best overall Stubbs result
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Results: a0.8_sm_2ms — Best Stubbs Result (0.007 °C/ping)")
rule(sl)

txb = sl.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.4))
tf = txb.text_frame; p = tf.paragraphs[0]
p.text = "α = 0.8  |  sm BW = 6 kHz (38.5–44.5 kHz)  |  T = 2 ms  |  H(f)^{−0.8} compensation"
p.font.size = Pt(13); p.font.color.rgb = GRAY

add_image(sl,
    f'{BASE}/recordings/20260710_031308_alpha_sweep_voltage_compensated/plots/a0.8_sm_2ms/stubbs_a0.8_sm_2ms_received.png',
    Inches(0.4), Inches(1.55), height=Inches(5.7))

add_image(sl,
    f'{BASE}/recordings/20260710_031308_alpha_sweep_voltage_compensated/plots/a0.8_sm_2ms/info_a0.8_sm_2ms.png',
    Inches(9.9), Inches(1.55), height=Inches(5.7))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12: Statistical histograms — bw_25kHz
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Statistical Distributions — bw_25kHz_comp (Autocorr + Stubbs)")
rule(sl)

# TOF hist (autocorr)
add_image(sl,
    f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/hist_tof_bw_25kHz_wide_2ms_comp.png',
    Inches(0.3), Inches(1.15), width=Inches(6.3))

# Stubbs hist
add_image(sl,
    f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/hist_stubbs_bw_25kHz_wide_2ms_comp.png',
    Inches(6.7), Inches(1.15), width=Inches(6.3))

# Labels
for x, txt in [(Inches(0.3), "Per-period autocorr  (N=1764, σ=0.861 μs → 0.084 °C/ping)"),
               (Inches(6.7), "Stubbs matched filter  (N=1665 eval, σ(Δτ)=0.655 μs → 0.064 °C/ping)")]:
    txb = sl.shapes.add_textbox(x, Inches(5.35), Inches(6.2), Inches(0.3))
    tf = txb.text_frame; p = tf.paragraphs[0]; p.text = txt
    p.font.size = Pt(10.5); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13: δ_peak histograms
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Fit Uncertainty δ_peak Distributions (DoF = 8, 11-pt Parabola)")
rule(sl)

for col, (wf_dir, label, caption) in enumerate([
    (f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp',
     'hist_dpeak_bw_25kHz_wide_2ms_comp.png', 'bw_25kHz_comp  (δ_peak ≈ 138 ns)'),
    (f'{BASE}/recordings/20260710_031308_alpha_sweep_voltage_compensated/plots/a0.3_lg_2ms',
     'hist_dpeak_a0.3_lg_2ms.png', 'a0.3_lg_2ms  (δ_peak ≈ 51 ns  — sharpest)'),
]):
    x = Inches(0.3 + col * 6.5)
    add_image(sl, f'{wf_dir}/{label}', x, Inches(1.2), width=Inches(6.2))
    txb = sl.shapes.add_textbox(x, Inches(5.4), Inches(6.2), Inches(0.3))
    tf = txb.text_frame; p = tf.paragraphs[0]; p.text = caption
    p.font.size = Pt(11); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

body_box(sl, "All valid waveforms: 0 periods clipped at 5σ — no outliers dragging distributions wide.",
         top=Inches(6.9), height=Inches(0.4), fontsize=12, color=GRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14: Δτ wandering — interesting plots
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Differential TOF Time Series  (Δτ = τ_return − τ_launch)")
rule(sl)

body_box(sl,
"""• Mean Δτ ≠ 0 reflects temperature offset between template epoch and evaluation window
• Increasing Δτ = cooling (slower sound);  decreasing Δτ = warming
• Slow drift (~1–2 μs over 30s) = real lab thermal fluctuation (~0.1–0.2 °C) — environment-limited, not instrument-limited
• σ(Δτ) per ping; uncertainty on mean temperature over 30s: σ/√N ≈ 0.009 μs → 0.0009 °C""",
left=Inches(0.5), top=Inches(1.15), width=Inches(5.5), height=Inches(2.5), fontsize=14)

# bw_25kHz Stubbs (shows nice drift)
add_image(sl,
    f'{BASE}/stubbs_full_analysis/stubbs_bw_25kHz_wide_2ms_comp_received.png',
    Inches(0.3), Inches(3.5), width=Inches(6.3))

# a0.8_sm_2ms (best result)
add_image(sl,
    f'{BASE}/stubbs_full_analysis/stubbs_a0.8_sm_2ms_received.png',
    Inches(6.8), Inches(1.15), width=Inches(6.3))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15: Δτ wandering cont. — another interesting case
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Δτ Time Series — Additional Examples")
rule(sl)

add_image(sl,
    f'{BASE}/stubbs_full_analysis_v2/stubbs_bw_25kHz_wide_2ms_comp_received.png',
    Inches(0.3), Inches(1.15), width=Inches(6.3))

add_image(sl,
    f'{BASE}/stubbs_full_analysis_v2/stubbs_a0.3_lg_2ms_received.png',
    Inches(6.8), Inches(1.15), width=Inches(6.3))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16: Guide / Supplemental — waveform naming
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
title_box(sl, "Supplemental: Waveform Naming Guide", fontsize=26)
rule(sl)
body_box(sl, """Bandwidth / frequency range
  bw_6kHz     6 kHz, symmetric around 41.5 kHz resonance
  bw_20kHz    20 kHz, 31.5–51.5 kHz
  bw_25kHz    25 kHz, 29–54 kHz (widest standard)
  hf_m15dB_asym   High-freq asymmetric; cutoff at −15 dB on the H(f) curve — extends further on low-freq side

Amplitude shaping
  comp          1/H(f) piezo compensation, 10% floor (max 20 dB boost)
  comp_light    1/H(f), 20% floor (max 14 dB boost) — less ring-down
  comp_rect     comp + rectangular window (no Tukey taper)
  diff          Diffraction-compensated: adds f/f₀ pre-emphasis on top of 1/H(f)
  flat          Constant amplitude — no compensation

Alpha sweep  (a{α}_{bw}_{dur})
  a0.2…a0.8   H(f)^{−α}: α=0 flat, α=1 full 1/H(f)
  sm / md / lg   6 / 12 / 20 kHz bandwidth

Duration: 2ms / 4ms  —  chirp duration before dead time (TBP = BW × T)

Voltage calibration (July 8–9)
  Calibrated per-waveform to 90% UltraMic full scale (2.15–8.64 Vpp).
  Recorded in 8th column of CHIRP_CONFIGS (chirp_sweep.py) and VOLTAGE_MAP (alpha_sweep.py).""",
fontsize=13)

# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════
out = f'{BASE}/BATPING_presentation_July2026.pptx'
prs.save(out)
print(f"\nSaved: {out}")
print(f"Slides: {len(prs.slides)}")
