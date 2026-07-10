#!/usr/bin/env python3
"""
Rebuild BATPING presentation with proper formatting:
- Larger text throughout
- LaTeX equations rendered via matplotlib → embedded PNGs
- Each figure properly sized (no upscaling, no overlaps)
- Breathing room on every slide
- More slides when needed
"""
import os, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = False  # use mathtext

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE    = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/siglent_project'
AGILENT = '/Users/cyrusurheim_1/Downloads/Stubbs_Stuff/Claude_Dir/agilent_project'
TMP     = tempfile.mkdtemp()

# ── Slide dimensions: 13.33 × 7.5 inches (16:9 widescreen) ──────────────────
SW, SH = Inches(13.33), Inches(7.5)
MARGIN = Inches(0.55)
CONTENT_W = Inches(12.23)

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY  = RGBColor(0x1a, 0x1a, 0x4e)
RED   = RGBColor(0xc0, 0x39, 0x2b)
DKGRAY= RGBColor(0x33, 0x33, 0x33)
GRAY  = RGBColor(0x55, 0x55, 0x55)
LTGRAY= RGBColor(0xaa, 0xaa, 0xaa)

prs = Presentation()
prs.slide_width  = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

# ── LaTeX rendering ────────────────────────────────────────────────────────────
def render_eq(latex, fontsize=22, dpi=300, fg='#1a1a4e', pad=0.15):
    """Render a LaTeX equation string → temp PNG path."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis('off')
    t = ax.text(0.5, 0.5, f'${latex}$',
                fontsize=fontsize, color=fg,
                ha='center', va='center', transform=ax.transAxes)
    # Measure and resize
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    w_in = bb.width  / dpi + 2*pad
    h_in = bb.height / dpi + 2*pad
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(w_in, 0.5), max(h_in, 0.3)))
    ax.axis('off')
    ax.text(0.5, 0.5, f'${latex}$', fontsize=fontsize, color=fg,
            ha='center', va='center', transform=ax.transAxes)
    fig.patch.set_alpha(0)
    path = os.path.join(TMP, f'eq_{abs(hash(latex))}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=pad/2)
    plt.close(fig)
    return path

def render_block(lines, fontsize=18, dpi=200, fg='#1a1a4e', width_in=10.0):
    """Render multiple lines (mix of text and $LaTeX$) to a PNG."""
    n = len(lines)
    h_in = n * (fontsize / 72.0) * 1.6 + 0.3
    fig, ax = plt.subplots(figsize=(width_in, h_in))
    ax.axis('off')
    y = 1.0 - 0.5 / n
    dy = 1.0 / n
    for line in lines:
        ax.text(0.0, y, line, fontsize=fontsize, color=fg,
                ha='left', va='center', transform=ax.transAxes,
                fontfamily='monospace' if line.startswith(' ') else 'sans-serif')
        y -= dy
    fig.patch.set_alpha(0)
    path = os.path.join(TMP, f'block_{abs(hash(str(lines)))}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0.05)
    plt.close(fig)
    return path

# ── Helper: add slide ──────────────────────────────────────────────────────────
def slide():
    return prs.slides.add_slide(BLANK)

def title(sl, text, y=Inches(0.28), size=34, color=NAVY):
    txb = sl.shapes.add_textbox(MARGIN, y, CONTENT_W, Inches(0.65))
    tf = txb.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]; p.text = text
    p.font.bold = True; p.font.size = Pt(size); p.font.color.rgb = color
    return txb

def rule(sl, y=Inches(1.0), color=NAVY):
    s = sl.shapes.add_shape(1, MARGIN, y, CONTENT_W, Pt(2))
    s.line.color.rgb = color; s.fill.background()

def txt(sl, text, left=MARGIN, top=Inches(1.1), width=CONTENT_W,
        height=Inches(5.9), size=18, color=DKGRAY, bold=False):
    txb = sl.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame; tf.word_wrap = True
    lines = text.strip().split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size); p.font.color.rgb = color
        p.font.bold = bold and (i == 0)

def img(sl, path, left, top, width=None, height=None):
    if not os.path.exists(path):
        print(f"  MISSING: {os.path.basename(path)}"); return None
    kw = {}
    if width:  kw['width']  = width
    if height: kw['height'] = height
    return sl.shapes.add_picture(path, left, top, **kw)

def caption(sl, text, y=Inches(7.1), size=11):
    txb = sl.shapes.add_textbox(MARGIN, y, CONTENT_W, Inches(0.3))
    tf = txb.text_frame; p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(size); p.font.color.rgb = LTGRAY
    p.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════════
# Pre-render LaTeX equations
# ════════════════════════════════════════════════════════════════════════════════
print("Rendering LaTeX equations...")
eq_fwhm    = render_eq(r'\mathrm{FWHM} = \frac{0.886}{B}', fontsize=28)
eq_sinc    = render_eq(r'y(\tau) = \mathrm{IFT}\!\left(|S(f)|^2\right) = B\cdot\mathrm{sinc}(B\tau)', fontsize=26)
eq_psd     = render_eq(r'|S(f)|^2 \approx \mathrm{rect}\!\left(\frac{f-f_0}{B}\right)', fontsize=26)
eq_sigma   = render_eq(r'\sigma_\mathrm{TOF} \approx \frac{\mathrm{FWHM}}{3.33 \times \mathrm{SNR}}', fontsize=26)
eq_sigma_T = render_eq(r'\sigma_T = \frac{\sigma_\mathrm{TOF}}{10.3\;\mu\mathrm{s}\cdot{^\circ\mathrm{C}^{-1}}}', fontsize=26)
eq_gauss   = render_eq(r'f_\theta(t)=A e^{-(t-t_c)^2/2\sigma^2}\cos(2\pi f_0 t+\phi)+\sum_k c_k g_k(t)', fontsize=22)
eq_alpha   = render_eq(r'A(f) \propto H(f)^{-\alpha},\quad \alpha\in[0,1]', fontsize=26)
eq_dpeak   = render_eq(r'\delta_\mathrm{peak} = \frac{\mathrm{RMS_{resid}}}{|a|\cdot\sqrt{\sum x^2}}', fontsize=26)
print("  Done.")

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
txb = sl.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(1.8))
tf = txb.text_frame
p = tf.paragraphs[0]; p.text = "BATPING"
p.font.size = Pt(60); p.font.bold = True
p.font.color.rgb = NAVY; p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "Bidirectional Acoustics Traveltime Pulser for INdex Gradients"
p2.font.size = Pt(24); p2.font.color.rgb = GRAY; p2.alignment = PP_ALIGN.CENTER
tf.add_paragraph()
p4 = tf.add_paragraph()
p4.text = "Cyrus Urheim  ·  Stubbs Group, Harvard  ·  July 2026"
p4.font.size = Pt(18); p4.font.color.rgb = LTGRAY; p4.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Introduction
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Introduction")
rule(sl)
txt(sl, """\
Statement of Purpose
  • LSST/Rubin dome & mirror seeing remains uncharacterized (Urbach et al. 2026)
  • Temperature fluctuations → index-of-refraction gradients → PSF broadening
  • BATPING: ultrasonic TOF measures the line-integral of temperature along the optical path
  • Advantage over optical methods: doesn't interfere with observations, no dome illumination needed

Precision Goal
  • Target: σ_T < 0.1 °C per ping
  • Requires σ_TOF < 1.02 μs  (2 m round-trip, 20 °C)
  • Best result to date: σ(Δτ) = 0.076 μs → 0.007 °C/ping

Anticipated Final Implementation
  • Winfield Hill transponder PCB: retransmits received burst after fixed delay → adaptive matched filter
  • Hexagonal piezo array (~40 kHz) + Dodotronic UltraMic 384k
  • Rubin dome crossing-path geometry; continuous operation during observations""",
size=17)

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Progress checklist (blank)
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Progress Since Last Meeting  [to fill in]")
rule(sl)
for i, y in enumerate([1.15, 1.75, 2.35, 2.95, 3.55, 4.15, 4.75, 5.35]):
    txb = sl.shapes.add_textbox(MARGIN, Inches(y), CONTENT_W, Inches(0.5))
    tf = txb.text_frame; p = tf.paragraphs[0]
    p.text = "☐"; p.font.size = Pt(20); p.font.color.rgb = LTGRAY

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Waveform Design I: Modified Gaussian (Bayesian opt)
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Waveform Design (I): Modified Gaussian — Bayesian Optimization")
rule(sl)
txt(sl, """\
Parametrization:
  Gaussian envelope × 40 kHz carrier + K sinusoidal compensation bursts

Objective (v1–v3):  minimize FWHM of ACF peak""",
    top=Inches(1.1), height=Inches(1.6), size=18)
img(sl, eq_gauss, Inches(1.5), Inches(2.75), height=Inches(0.7))
txt(sl, """\
Results:
  v3  (N=1000 iter): 38% FWHM improvement  443 → 275 μs
  Marginal gain v2→v3 only 7% — parametrization bottleneck
  Bottleneck: ring-down of piezo (τ ≈ Q/πf₀ ≈ 180 μs) sets floor""",
    top=Inches(3.55), height=Inches(1.8), size=18)
img(sl, f'{AGILENT}/bayesian_optimization/waveform_comparison_overlay.png',
    Inches(7.2), Inches(1.1), width=Inches(5.8))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Waveform Design I cont: Barker codes
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Waveform Design (I): Barker-7 Codes")
rule(sl)
txt(sl, """\
Barker-7:  binary phase-shift keying (BPSK) onto 40 kHz carrier

Properties:
  • Autocorrelation peak = 7 (# chips), sidelobes ∈ {−1, 0, +1}
  • FWHM ∝ T_chip = N_cyc / f_carrier  (observed: linear, as expected)
  • Measured sidelobes: −12 to −14 dB  (theoretical: −16.9 dB)

Limitation: FWHM floor same as optimized Gaussian at matched chip size
  → both approaches limited by piezo ring-down, not sequence length""",
    top=Inches(1.1), width=Inches(6.0), height=Inches(4.5), size=17)
img(sl, f'{BASE}/barker7_32cyc_v5.png',
    Inches(6.4), Inches(1.1), width=Inches(6.6))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Waveform Design II: LFM Chirp Rationale
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Waveform Design (II): LFM Chirped Waveforms")
rule(sl)

txt(sl, "Why a chirp?  Sweeps f_start → f_stop over duration T — a delta function spread out in time.",
    top=Inches(1.1), height=Inches(0.5), size=18)

txt(sl, "Nearly flat power spectrum over bandwidth B:",
    top=Inches(1.65), height=Inches(0.4), size=17, color=GRAY)
img(sl, eq_psd, Inches(2.0), Inches(2.1), height=Inches(0.7))

txt(sl, "Autocorrelation peak shape (Wiener-Khinchin):",
    top=Inches(2.9), height=Inches(0.4), size=17, color=GRAY)
img(sl, eq_sinc, Inches(1.0), Inches(3.35), height=Inches(0.7))

txt(sl, "Expected −3 dB width:",
    top=Inches(4.15), height=Inches(0.4), size=17, color=GRAY)
img(sl, eq_fwhm, Inches(3.5), Inches(4.6), height=Inches(0.7))

txt(sl, "Key caveat — SNR also matters:",
    top=Inches(5.45), height=Inches(0.4), size=17, color=GRAY)
img(sl, eq_sigma, Inches(1.5), Inches(5.9), height=Inches(0.65))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — LFM Chirp: Amplitude compensation & α
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "LFM Chirp: Amplitude Compensation")
rule(sl)
txt(sl, """\
flat       Constant amplitude — simple, surprisingly competitive
comp       1/H(f), 10% floor (max 20 dB boost) — flattens acoustic output spectrum
comp_light 1/H(f), 20% floor (max 14 dB boost) — less ring-down excitation
α sweep    H(f)^{−α}  continuously interpolates flat (α=0) → full inversion (α=1)""",
    top=Inches(1.1), height=Inches(2.2), size=17)
img(sl, eq_alpha, Inches(2.5), Inches(3.4), height=Inches(0.7))
txt(sl, """\
Ring-down / double-dagger problem:
  • Aggressive compensation drives off-resonance frequencies hard
  • Piezo ring-down at f₀ creates two temporal clusters → double peak in autocorr
  • Root cause: group delay dispersion — two-way peak delay ≈ 343 μs  (Q = 22.8)
  • Practical limit: 10–20% compensation floor to avoid ring-down artifacts""",
    top=Inches(4.2), height=Inches(2.5), size=17)

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Chirp BW sweep (standalone figure)
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "LFM Chirp: FWHM and σ_TOF vs Bandwidth Sweep")
rule(sl)
img(sl, f'{BASE}/chirp_bw_sweep_comp_v2.png',
    Inches(0.8), Inches(1.15), width=Inches(11.7))
caption(sl, ("FWHM saturates at ~95 μs for BW ≥ 13 kHz (noise/multipath floor, not piezo limit).  "
             "σ_TOF first crosses 0.1°C at BW = 12 kHz.  "
             "FWHM tracks 1/BW; σ_TOF does not — SNR matters as much as bandwidth."))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Piezo Transfer Function H(f)  (standalone)
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Piezoelectric Transducer: Transfer Function H(f)")
rule(sl)
img(sl, f'{BASE}/piezo_characterization.png',
    Inches(1.8), Inches(1.1), width=Inches(9.7))
caption(sl, ("Stepped-sine characterization, 20–65 kHz, 50 Hz steps  ·  "
             "f₀ = 41.3 kHz  ·  Q ≈ 22.8  ·  −3 dB BW = 2.95 kHz  ·  "
             "asymmetric: extends further on low-freq side"))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Metrics Definitions
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Measurement Metrics")
rule(sl)

txt(sl, "FWHM_corr   Full width at half max of TOF peak in the per-period autocorrelation.",
    top=Inches(1.1), height=Inches(0.4), size=17)
txt(sl, "              11-pt least-squares parabolic fit (DoF = 8) → sub-sample peak position.",
    top=Inches(1.5), height=Inches(0.4), size=17, color=GRAY)

txt(sl, "δ_peak      1σ fit uncertainty from parabola residuals:",
    top=Inches(2.0), height=Inches(0.4), size=17)
img(sl, eq_dpeak, Inches(2.5), Inches(2.4), height=Inches(0.65))

txt(sl, "σ_TOF       Std. dev. of per-period TOF across N pings  (key temperature metric):",
    top=Inches(3.15), height=Inches(0.4), size=17)
img(sl, eq_sigma_T, Inches(2.5), Inches(3.55), height=Inches(0.65))
txt(sl, "              Target: σ_TOF < 1.02 μs per ping  →  0.1 °C",
    top=Inches(4.3), height=Inches(0.4), size=16, color=GRAY)

txt(sl, "σ(Δτ)       Stubbs pipeline: σ(τ_return − τ_launch).  Cancels alignment jitter.",
    top=Inches(4.8), height=Inches(0.4), size=17)
txt(sl, "SNR (CR)    Cramér-Rao implied SNR = FWHM / (3.33 × σ_TOF).",
    top=Inches(5.35), height=Inches(0.4), size=17)
txt(sl, "              meas/CR ≈ 1–3: instrument-limited.  meas/CR >> 3: environment-limited.",
    top=Inches(5.75), height=Inches(0.4), size=16, color=GRAY)

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Methods: Autocorr + Stubbs overview (text)
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Analysis Methods")
rule(sl)
txt(sl, """\
Per-period autocorrelation  (fast, online)
  • Autocorrelate each received period with itself
  • TOF peak at lag ≈ 5.7 ms (2 m round-trip)
  • FWHM and peak position extracted with 11-pt parabolic fit
  • σ_TOF from scatter of per-period peak positions over N pings

Stubbs matched-filter pipeline  (30 s, offline)
  • Adaptive period slicer: finds actual incident peaks (no fixed clock boundaries)
    → eliminates SDG ↔ UltraMic clock drift issue (3.5 μs/period over 30 s)
  • Co-add 100 drift-corrected periods → T_launch, T_return  (√100 × SNR gain)
  • Evaluate: xcorr each period with templates → τ_launch, τ_return
  • Δτ = τ_R − τ_L  (differential TOF, cancels common-mode jitter)
  • τ_launch ≈ 0 is the alignment quality diagnostic
  • σ(Δτ) is the temperature precision""",
    top=Inches(1.1), height=Inches(5.8), size=17)

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Alignment failure modes (ppp_xcorr)  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Stubbs Pipeline: Alignment Failure Modes (Fixed)")
rule(sl)
img(sl, f'{BASE}/ppp_xcorr_diagnostic.png',
    Inches(0.55), Inches(1.1), width=Inches(12.2))
caption(sl, ("Three error modes in the burst-position alignment: "
             "wrong peak (burst outside gate), edge cut-off, next-period bleed. "
             "All fixed via global-max ±TOF pairing with interior-only search."))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Fixed window detection  (period_comparison_all3)  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Stubbs Pipeline: Corrected Window Detection")
rule(sl)
img(sl, f'{BASE}/period_comparison_all3.png',
    Inches(0.55), Inches(1.1), width=Inches(12.2))
caption(sl, ("bw_25kHz (top): clean incident and reflected windows.  "
             "hf_comp_rect (middle): correctly identified despite narrow BW.  "
             "bw_6kHz_flat (bottom): incident visible with new mic position."))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Results: bw_25kHz recording plot  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Results — bw_25kHz_comp: Input · Received · Autocorr")
rule(sl)
txt(sl, "BW = 25 kHz  |  T = 2 ms  |  TBP = 50  |  1/H(f) comp, 10% floor  |  6.22 Vpp",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/bw_25kHz_wide_2ms_comp.png',
    Inches(0.55), Inches(1.5), width=Inches(12.2))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Results: bw_25kHz Stubbs plot  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Results — bw_25kHz_comp: Stubbs Matched Filter")
rule(sl)
txt(sl, "σ(Δτ) = 0.655 μs → 0.064 °C/ping  |  N = 1665 eval periods",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/stubbs_bw_25kHz_wide_2ms_comp_received.png',
    Inches(1.2), Inches(1.5), height=Inches(5.7))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Results: a0.8_sm_2ms — best result  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Results — a0.8_sm_2ms: Best Overall  (σ = 0.076 μs → 0.007 °C/ping)")
rule(sl)
txt(sl, "α = 0.8  |  BW = 6 kHz (38.5–44.5 kHz)  |  T = 2 ms  |  H(f)^{−0.8}",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_031308_alpha_sweep_voltage_compensated/plots/a0.8_sm_2ms/stubbs_a0.8_sm_2ms_received.png',
    Inches(1.2), Inches(1.5), height=Inches(5.7))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — Statistical histograms: TOF (autocorr)  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "TOF Distribution — bw_25kHz_comp (Per-period Autocorr)")
rule(sl)
txt(sl, "N = 1764 periods  |  σ_TOF = 0.861 μs  |  0 outliers at 5σ  |  near-Gaussian",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/hist_tof_bw_25kHz_wide_2ms_comp.png',
    Inches(2.0), Inches(1.5), width=Inches(9.3))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 18 — Statistical histograms: Stubbs Δτ  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Δτ Distribution — bw_25kHz_comp (Stubbs Pipeline)")
rule(sl)
txt(sl, "N = 1665 eval  |  σ(Δτ) = 0.655 μs  |  τ_launch, τ_return, Δτ = τ_R − τ_L",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_030748_chirp_sweep/plots/bw_25kHz_wide_2ms_comp/hist_stubbs_bw_25kHz_wide_2ms_comp.png',
    Inches(0.55), Inches(1.5), width=Inches(12.2))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 19 — δ_peak histogram  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Fit Uncertainty δ_peak  (DoF = 8, 11-pt Parabola)")
rule(sl)
txt(sl, "a0.3_lg_2ms: δ_peak ≈ 51 ns (sharpest — 20 kHz BW gives tight peaks)",
    top=Inches(1.05), height=Inches(0.4), size=16, color=GRAY)
img(sl, f'{BASE}/recordings/20260710_031308_alpha_sweep_voltage_compensated/plots/a0.3_lg_2ms/hist_dpeak_a0.3_lg_2ms.png',
    Inches(2.0), Inches(1.5), width=Inches(9.3))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 20 — Δτ Wandering: bw_25kHz  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Δτ Time Series — bw_25kHz_comp")
rule(sl)
txt(sl, ("Increasing Δτ = cooling (longer TOF).  "
         "Slow drift ≈ 1 μs over 30 s ≈ real lab thermal fluctuation (~0.1 °C).  "
         "Environment-limited, not instrument-limited."),
    top=Inches(1.05), height=Inches(0.5), size=16, color=GRAY)
img(sl, f'{BASE}/stubbs_full_analysis/stubbs_bw_25kHz_wide_2ms_comp_received.png',
    Inches(1.2), Inches(1.6), height=Inches(5.6))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 21 — Δτ Wandering: a0.8_sm_2ms  [standalone]
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Δτ Time Series — a0.8_sm_2ms (Best Result)")
rule(sl)
txt(sl, ("σ(Δτ) = 0.076 μs → 0.007 °C/ping.  "
         "Note: τ_launch σ ≈ 0.61 μs >> τ_return σ ≈ 0.57 μs — "
         "differential cancels common-mode jitter."),
    top=Inches(1.05), height=Inches(0.5), size=16, color=GRAY)
img(sl, f'{BASE}/stubbs_full_analysis/stubbs_a0.8_sm_2ms_received.png',
    Inches(1.2), Inches(1.6), height=Inches(5.6))

# ════════════════════════════════════════════════════════════════════════════════
# SLIDE 22 — Supplemental: Waveform Naming Guide
# ════════════════════════════════════════════════════════════════════════════════
sl = slide()
title(sl, "Supplemental: Waveform Naming Guide", size=28)
rule(sl)
txt(sl, """\
Bandwidth / frequency range
  bw_6kHz        6 kHz, symmetric around 41.5 kHz
  bw_20kHz       20 kHz, 31.5–51.5 kHz
  bw_25kHz       25 kHz, 29–54 kHz  (widest)
  hf_m15dB_asym  High-freq asymmetric; cutoff at −15 dB on H(f); extends further on low-freq side
  sm / md / lg   6 / 12 / 20 kHz  (alpha sweep)

Amplitude shaping
  flat         Constant amplitude — no compensation
  comp         1/H(f), 10% floor  (max 20 dB boost)
  comp_light   1/H(f), 20% floor  (max 14 dB boost) — less ring-down
  comp_rect    comp + rectangular window  (no Tukey taper)
  diff         Diffraction-compensated: f/f₀ pre-emphasis on top of 1/H(f)
  a0.2…a0.8   H(f)^{−α}: α=0 flat, α=1 full 1/H(f)

Duration: 2ms / 4ms  — chirp duration before dead time  (TBP = BW × T)

Voltage calibration (July 8–9)
  Per-waveform Vpp calibrated to 90% UltraMic FS.
  Stored in CHIRP_CONFIGS (chirp_sweep.py) and VOLTAGE_MAP (alpha_sweep.py).""",
    top=Inches(1.1), size=15)

# ════════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════════
out = f'{BASE}/BATPING_presentation_July2026.pptx'
prs.save(out)
print(f"\nSaved: {out}")
print(f"Total slides: {len(prs.slides)}")
for i, sl in enumerate(prs.slides):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.paragraphs[0].text:
            print(f"  {i+1:2d}. {sh.text_frame.paragraphs[0].text[:65]}"); break

import shutil; shutil.rmtree(TMP)
