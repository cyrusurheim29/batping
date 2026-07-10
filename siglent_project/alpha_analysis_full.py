#!/usr/bin/env python3
"""
Full analysis of new voltage-calibrated alpha sweep recordings.
For each waveform computes and adds to info panel:
  - Autocorr: sigma_TOF (3s), FWHM, delta_peak (fit uncertainty), actual SNR
  - Stubbs:   sigma_Dtau, sigma_tL, sigma_tR, mean_fit_sigma
  - Comparison: Stubbs/autocorr ratio
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont
import chirp_sweep as cs
import chirp_stubbs_analysis as st

BASE   = 'recordings/20260710_031308_alpha_sweep/raw'
PLOTS  = 'recordings/20260710_031308_alpha_sweep/plots'
OUTDIR = 'recordings/20260710_031308_alpha_sweep/plots_annotated'
SDIR   = 'recordings/20260710_031308_alpha_sweep/stubbs'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(SDIR,   exist_ok=True)

SDG_SR=400000; SDG_MAX=16384
def rr(dur): return SDG_SR/min(int((dur+15)/1000*SDG_SR),SDG_MAX)

SR = 384000
def hp(s): return sosfiltfilt(butter(8,30000/(SR/2),btype='high',output='sos'),s)

# Table from alpha sweep printout (sigma and FWHM from 3s autocorr)
TABLE = {
    'a0.4_sm_4ms':(0.306,0.0298,266), 'a0.4_md_4ms':(0.353,0.0343,193),
    'a0.6_lg_2ms':(0.368,0.0358,159), 'a0.2_md_4ms':(0.389,0.0377,216),
    'a0.6_md_4ms':(0.415,0.0403,177), 'a0.3_md_2ms':(0.427,0.0415,208),
    'a0.3_sm_2ms':(0.439,0.0426,276), 'a0.4_lg_2ms':(0.448,0.0435,182),
    'a0.3_lg_2ms':(0.449,0.0436,198), 'a0.2_sm_2ms':(0.458,0.0444,284),
    'a0.4_sm_2ms':(0.470,0.0457,271), 'a0.5_md_2ms':(0.470,0.0457,193),
    'a0.5_sm_4ms':(0.490,0.0475,258), 'a0.3_sm_4ms':(0.500,0.0486,271),
    'a0.8_sm_4ms':(0.573,0.0556,247), 'a0.6_sm_4ms':(0.603,0.0586,255),
    'a0.8_md_4ms':(0.612,0.0594,180), 'a0.5_lg_2ms':(0.637,0.0619,169),
    'a0.2_sm_4ms':(0.670,0.0651,279), 'a0.8_md_2ms':(0.699,0.0678,182),
    'a0.5_lg_4ms':(0.701,0.0681,172), 'a0.5_md_4ms':(0.708,0.0687,185),
    'a0.4_lg_4ms':(0.721,0.0700,185), 'a0.2_md_2ms':(0.725,0.0704,221),
    'a0.3_md_4ms':(0.750,0.0729,206), 'a0.6_sm_2ms':(0.788,0.0765,263),
    'a0.6_md_2ms':(0.792,0.0769,188), 'a0.4_md_2ms':(0.853,0.0828,198),
    'a0.8_sm_2ms':(0.927,0.0900,255), 'a0.5_sm_2ms':(0.928,0.0901,266),
    'a0.2_lg_4ms':(0.934,0.0907,214), 'a0.2_lg_2ms':(0.984,0.0956,214),
    'a0.3_lg_4ms':(0.994,0.0965,198), 'a0.8_lg_2ms':(1.189,0.1154,154),
    'a0.6_lg_4ms':(1.427,0.1385,164), 'a0.8_lg_4ms':(2.115,0.2053,159),
}

def actual_snr(sig_raw, rr_hz):
    """Signal RMS in reflected burst / noise RMS before burst."""
    sig = hp(sig_raw)
    pn = round(SR/rr_hz)
    # Use 3rd period (settled)
    i0 = min(3, len(sig)//pn-1)
    seg = sig[i0*pn:(i0+1)*pn]
    t_ms = np.arange(len(seg))/SR*1000
    # Auto-detect burst region via peak
    env = np.abs(hilbert(seg))
    peak_t = t_ms[np.argmax(env)]
    # Reflected: ~peak_t ± 4ms (generous)
    ref_lo = max(0, int((peak_t-3)/1000*SR))
    ref_hi = min(len(seg), int((peak_t+3)/1000*SR))
    sig_rms = np.sqrt(np.mean(seg[ref_lo:ref_hi]**2)) if ref_hi>ref_lo else 1e-6
    # Noise: first 2ms of period
    noise_hi = int(0.002*SR)
    noise_rms = np.sqrt(np.mean(seg[:noise_hi]**2)) if noise_hi>0 else 1e-6
    snr_lin = sig_rms / (noise_rms + 1e-12)
    return snr_lin, 20*np.log10(snr_lin+1e-12)

results = {}
labels_sorted = sorted(TABLE.keys(),
                       key=lambda k: TABLE[k][0])  # sort by sigma_TOF

for label in labels_sorted:
    sigma_ac, degc_ac, fwhm = TABLE[label]
    dur_ms = 4.0 if '4ms' in label else 2.0
    rep = rr(dur_ms)
    wav = f'{BASE}/{label}_received.wav'
    if not os.path.exists(wav):
        print(f'  SKIP {label}'); continue

    print(f'\n--- {label} ---', flush=True)
    fs2, raw = wavfile.read(wav)
    sig_raw = raw.astype(np.float64)/32768.0

    # 1. Actual SNR
    snr_lin, snr_db = actual_snr(sig_raw, rep)

    # 2. Per-period autocorr fit sigma (3s, recompute to get delta_peak)
    _, sigma_ac2, n_pp, fit_sig_ac = cs.per_period_sigma_tof(
        sig_raw, rep, analysis_duration_s=3.0)
    delta_peak_ac = fit_sig_ac  # mean 11-pt fit uncertainty (μs)

    # 3. Stubbs analysis (full 30s with adaptive slicer)
    try:
        r = st.analyze(wav, rep_rate=rep, output_dir=SDIR)
        sigma_stubbs = r['std_delta_tau_us']
        sigma_tl     = r['tau_launch'].std()
        sigma_tr     = r['tau_return'].std()
        fit_sig_st   = r.get('mean_fit_sigma_us', None)
        n_eval       = r['n_eval']
    except Exception as e:
        print(f'  Stubbs FAILED: {e}')
        sigma_stubbs = fit_sig_st = None
        sigma_tl = sigma_tr = n_eval = None

    results[label] = dict(
        sigma_ac=sigma_ac, degc_ac=degc_ac, fwhm=fwhm,
        snr_lin=snr_lin, snr_db=snr_db,
        delta_peak_ac=delta_peak_ac,
        sigma_stubbs=sigma_stubbs, sigma_tl=sigma_tl, sigma_tr=sigma_tr,
        fit_sig_stubbs=fit_sig_st, n_eval=n_eval,
    )
    s = f"  autocorr σ={sigma_ac:.3f}μs  δpeak={delta_peak_ac*1000:.0f}ns  SNR={snr_db:.1f}dB"
    if sigma_stubbs:
        s += f"  |  Stubbs σ={sigma_stubbs:.3f}μs  σ_tL={sigma_tl:.3f}  σ_tR={sigma_tr:.3f}"
        if fit_sig_st:
            s += f"  δpeak_st={fit_sig_st*1000:.0f}ns"
    print(s)

# ── Update info panels ────────────────────────────────────────────────────────
try:
    font_b = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    font   = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    font_v = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    font_s = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
except:
    font_b = font = font_v = font_s = ImageFont.load_default()

PANEL_W = 310; PAD = 10

for label, r in results.items():
    png = os.path.join(PLOTS, f'{label}.png')
    if not os.path.exists(png): continue
    alpha = label.split('_')[0][1:]
    bw_tag = label.split('_')[1]
    dur = label.split('_')[2]

    img = Image.open(png).convert('RGB')
    w, h = img.size
    new_img = Image.new('RGB', (w+PANEL_W, h), (245,245,245))
    new_img.paste(img, (0,0))
    draw = ImageDraw.Draw(new_img)
    draw.rectangle([(w,0),(w+PANEL_W,h)], fill=(245,245,245))
    draw.line([(w,0),(w,h)], fill=(170,170,170), width=2)

    snr_implied = r['fwhm']/(3.33*r['sigma_ac'])
    ratio = r['sigma_ac']/r['sigma_stubbs'] if r['sigma_stubbs'] else None

    sections = [
        # Header
        [('bold', f"α={alpha}  BW={bw_tag}  {dur}")],
        # Autocorr section
        [('head', '─── Autocorr (3s) ───'),
         ('kv',   'σ_TOF',    f"{r['sigma_ac']:.3f} μs", True),
         ('kv',   '°C/ping',  f"{r['degc_ac']:.4f}", True),
         ('kv',   'FWHM',     f"{r['fwhm']} μs", False),
         ('kv',   'δ_peak',   f"{r['delta_peak_ac']*1000:.0f} ns  (DoF=8)", False),
         ('kv',   'SNR (sig)',f"{r['snr_db']:.1f} dB  ({r['snr_lin']:.0f}×)", False),
         ('kv',   'SNR (CR)', f"{snr_implied:.0f}×  ({20*np.log10(snr_implied):.1f} dB)", False)],
        # Stubbs section
        [('head', '─── Stubbs (30s) ───'),
         ('kv',   'σ(Δτ)',    f"{r['sigma_stubbs']:.3f} μs" if r['sigma_stubbs'] else 'N/A', True),
         ('kv',   '°C/ping',  f"{r['sigma_stubbs']/10.3:.4f}" if r['sigma_stubbs'] else 'N/A', True),
         ('kv',   'σ(τ_L)',   f"{r['sigma_tl']:.3f} μs" if r['sigma_tl'] else 'N/A', False),
         ('kv',   'σ(τ_R)',   f"{r['sigma_tr']:.3f} μs" if r['sigma_tr'] else 'N/A', False),
         ('kv',   'δ_peak',   f"{r['fit_sig_stubbs']*1000:.0f} ns" if r['fit_sig_stubbs'] else 'N/A', False),
         ('kv',   'N_eval',   f"{r['n_eval']}" if r['n_eval'] else 'N/A', False)],
        # Comparison
        [('head', '─── Comparison ───'),
         ('kv',   'AC/Stubbs', f"{ratio:.2f}×" if ratio else 'N/A',
          False if not ratio else ratio > 1.5)],
    ]

    y = PAD
    for section in sections:
        for item in section:
            kind = item[0]
            if kind == 'bold':
                draw.text((w+PAD, y), item[1], font=font_b, fill=(20,20,20)); y+=22
            elif kind == 'head':
                draw.text((w+PAD, y), item[1], font=font_s, fill=(100,100,140)); y+=17
            elif kind == 'kv':
                _, key, val, is_metric = item
                draw.text((w+PAD, y), f'{key}:', font=font_s, fill=(80,80,80)); y+=15
                col = (180,30,30) if is_metric else (20,20,20)
                draw.text((w+PAD+10, y), val, font=font_v, fill=col); y+=22
        y += 4

    new_img.save(os.path.join(OUTDIR, f'{label}.png'), dpi=(150,150))

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*90}')
print(f'{"Label":<16} {"AC σ μs":>9} {"Stubbs σ":>9} {"AC/St":>7} '
      f'{"δpeak_ac ns":>12} {"δpeak_st ns":>12} {"SNR_sig dB":>11}')
print('-'*90)
for label in labels_sorted:
    if label not in results: continue
    r = results[label]
    ratio = r['sigma_ac']/r['sigma_stubbs'] if r['sigma_stubbs'] else float('nan')
    dps = r['fit_sig_stubbs']*1000 if r['fit_sig_stubbs'] else float('nan')
    print(f"  {label:<14} {r['sigma_ac']:>9.3f} "
          f"{r['sigma_stubbs']:>9.3f} {ratio:>7.2f}x "
          f"{r['delta_peak_ac']*1000:>12.0f} {dps:>12.0f} "
          f"{r['snr_db']:>11.1f}")

# ── Comparison scatter plot ───────────────────────────────────────────────────
valid = [(r['sigma_ac'], r['sigma_stubbs'], label)
         for label,r in results.items()
         if r['sigma_stubbs'] and r['sigma_stubbs'] < 5]
if valid:
    ac_vals, st_vals, lbls = zip(*valid)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(ac_vals, st_vals, s=40, alpha=0.8)
    for x,y_,l in zip(ac_vals, st_vals, lbls):
        ax.annotate(l, (x,y_), fontsize=6, alpha=0.7,
                    xytext=(3,3), textcoords='offset points')
    lim = max(max(ac_vals), max(st_vals))*1.1
    ax.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.5,label='y=x (methods equal)')
    ax.plot([0,lim],[0,lim/2],'C1--',lw=1,alpha=0.4,label='Stubbs 2× better')
    ax.set_xlabel('Autocorr σ_TOF (μs)'); ax.set_ylabel('Stubbs σ(Δτ) (μs)')
    ax.set_title('Autocorr vs Stubbs temperature precision\n(points below y=x → Stubbs is better)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/comparison_ac_vs_stubbs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot.")
print('\nDone.')
