#!/usr/bin/env python3
"""
Generate clean standalone info panel PNGs for each alpha sweep waveform.
Saved as plots/info_{label}.png alongside the existing recording plots.
Fixes SNR: uses tail of period (post-reflected quiet region) as noise floor.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.io import wavfile
import chirp_sweep as cs
import chirp_stubbs_analysis as st

BASE  = 'recordings/20260710_031308_alpha_sweep/raw'
PLOTS = 'recordings/20260710_031308_alpha_sweep/plots'

SR = 384000
SDG_SR=400000; SDG_MAX=16384
def rr(d): return SDG_SR/min(int((d+15)/1000*SDG_SR),SDG_MAX)
def hp(s): return sosfiltfilt(butter(8,30000/(SR/2),btype='high',output='sos'),s)

# Collect all analysis results from Stubbs log + recompute autocorr
LOG = open('/tmp/alpha_full.log').read()
pattern = (r'--- (a[\d.]+_\w+_\w+) ---.*?'
           r'autocorr σ=([\d.]+)μs.*?δpeak=([\d.]+)ns.*?SNR=([-\d.]+)dB'
           r'(?:.*?Stubbs σ=([\d.]+)μs.*?σ_tL=([\d.]+).*?σ_tR=([\d.]+).*?δpeak_st=([\d.]+)ns)?')
log_results = {}
for m in re.finditer(pattern, LOG, re.DOTALL):
    label = m.group(1)
    log_results[label] = dict(
        sigma_ac=float(m.group(2)), delta_peak_ac=float(m.group(3)),
        sigma_stubbs=float(m.group(5)) if m.group(5) else None,
        sigma_tl=float(m.group(6)) if m.group(6) else None,
        sigma_tr=float(m.group(7)) if m.group(7) else None,
        delta_peak_st=float(m.group(8)) if m.group(8) else None,
    )

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

def compute_snr(wav_path, rep_hz):
    """SNR = signal RMS in reflected burst / RMS in post-reflected quiet region."""
    fs2, raw = wavfile.read(wav_path)
    sig = hp(raw.astype(np.float64)/32768.0)
    pn = round(fs2/rep_hz)
    i0 = min(5, len(sig)//pn-1)
    seg = sig[i0*pn:(i0+1)*pn]
    t_ms = np.arange(len(seg))/fs2*1000
    env = np.abs(hilbert(seg))
    # Reflected burst: peak region ±3ms around max
    pk_t = t_ms[np.argmax(env)]
    r_lo = max(0, int((pk_t-3)/1000*fs2))
    r_hi = min(len(seg), int((pk_t+3)/1000*fs2))
    sig_rms = np.sqrt(np.mean(seg[r_lo:r_hi]**2))
    # Quiet region: last 3ms of period (after all echoes)
    q_lo = max(0, len(seg)-int(0.003*fs2))
    noise_rms = np.sqrt(np.mean(seg[q_lo:]**2)) if len(seg)-q_lo>10 else 1e-6
    snr_lin = sig_rms/(noise_rms+1e-12)
    return snr_lin, 20*np.log10(snr_lin+1e-12)

def make_info_panel(label, sigma_ac, degc_ac, fwhm,
                    delta_peak_ac, snr_lin, snr_db, snr_cr_lin,
                    sigma_stubbs, sigma_tl, sigma_tr, delta_peak_st):
    """Render a clean info panel as a matplotlib figure."""
    alpha = label.split('_')[0][1:]
    bw_tag = label.split('_')[1]
    dur = label.split('_')[2]
    bw_hz = {'sm':'6 kHz','md':'12 kHz','lg':'20 kHz'}.get(bw_tag, bw_tag)

    fig = plt.figure(figsize=(4.2, 5.8), facecolor='#f8f8f8')
    ax = fig.add_axes([0,0,1,1]); ax.axis('off')
    ax.set_facecolor('#f8f8f8')

    y = 0.97; lh = 0.064; lh_s = 0.050

    def txt(x, y_, s, **kw):
        ax.text(x, y_, s, transform=ax.transAxes,
                va='top', fontsize=kw.pop('fs',9.5), **kw)

    # Header
    txt(0.5, y, label, fs=12, fontweight='bold', ha='center',
        color='#1a1a2e'); y -= lh*1.1
    txt(0.5, y, f'α={alpha}   BW={bw_hz}   T={dur}', fs=9,
        ha='center', color='#555'); y -= lh*1.4

    # Divider
    ax.axhline(y, color='#aaa', lw=0.8, xmin=0.05, xmax=0.95)
    y -= lh*0.5

    # ── Autocorrelation section ───────────────────────────────────────────────
    txt(0.05, y, 'Per-period autocorr  (3 s, ~156 periods)', fs=8.5,
        color='#2c5f8a', fontweight='bold'); y -= lh*0.9

    rows_ac = [
        ('σ_TOF',         f'{sigma_ac:.3f} μs',    True),
        ('°C / ping',     f'{degc_ac:.4f}',          True),
        ('FWHM',          f'{fwhm} μs',              False),
        ('δ_peak (DoF=8)',f'{delta_peak_ac:.0f} ns', False),
        ('SNR (signal)',  f'{snr_db:.1f} dB  ({snr_lin:.0f}×)', False),
        ('SNR (CR)',      f'{snr_cr_lin:.0f}×  ({20*np.log10(snr_cr_lin):.1f} dB)', False),
    ]
    for key, val, red in rows_ac:
        col = '#c0392b' if red else '#333'
        txt(0.08, y, f'{key}', fs=8, color='#666'); y -= lh_s
        txt(0.12, y, val, fs=9, color=col, fontweight='bold' if red else 'normal'); y -= lh*0.85

    y -= lh*0.3
    ax.axhline(y, color='#ccc', lw=0.6, xmin=0.05, xmax=0.95); y -= lh*0.4

    # ── Stubbs section ────────────────────────────────────────────────────────
    txt(0.05, y, 'Stubbs matched filter  (30 s)', fs=8.5,
        color='#27612e', fontweight='bold'); y -= lh*0.9

    if sigma_stubbs and sigma_stubbs < 10:
        rows_st = [
            ('σ(Δτ)',          f'{sigma_stubbs:.3f} μs',     True),
            ('°C / ping',      f'{sigma_stubbs/10.3:.4f}',    True),
            ('σ(τ_launch)',    f'{sigma_tl:.3f} μs' if sigma_tl else 'N/A', False),
            ('σ(τ_return)',    f'{sigma_tr:.3f} μs' if sigma_tr else 'N/A', False),
            ('δ_peak (DoF=8)',f'{delta_peak_st:.0f} ns' if delta_peak_st else 'N/A', False),
        ]
        for key, val, red in rows_st:
            col = '#c0392b' if red else '#333'
            txt(0.08, y, f'{key}', fs=8, color='#666'); y -= lh_s
            txt(0.12, y, val, fs=9, color=col,
                fontweight='bold' if red else 'normal'); y -= lh*0.85

        # Comparison
        y -= lh*0.2
        ratio = sigma_ac/sigma_stubbs
        if ratio > 1.5:
            comp_str = f'Stubbs {ratio:.1f}× better'
            comp_col = '#27612e'
        elif ratio < 0.67:
            comp_str = f'Autocorr {1/ratio:.1f}× better'
            comp_col = '#2c5f8a'
        else:
            comp_str = 'Methods comparable'
            comp_col = '#666'
        ax.axhline(y, color='#ccc', lw=0.6, xmin=0.05, xmax=0.95); y -= lh*0.5
        txt(0.5, y, comp_str, fs=9, ha='center', color=comp_col, fontweight='bold')
    else:
        status = 'Stubbs failed (4ms period detection bug)' if '4ms' in label else 'Stubbs failed'
        txt(0.5, y, status, fs=8.5, ha='center', color='#999', style='italic')

    plt.savefig(os.path.join(PLOTS, f'info_{label}.png'),
                dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    plt.close()

# ── Generate all panels ───────────────────────────────────────────────────────
for label in sorted(TABLE):
    sigma_ac, degc_ac, fwhm = TABLE[label]
    dur_ms = 4.0 if '4ms' in label else 2.0
    wav = f'{BASE}/{label}_received.wav'
    if not os.path.exists(wav): print(f'SKIP {label}'); continue

    snr_lin, snr_db = compute_snr(wav, rr(dur_ms))
    snr_cr_lin = fwhm/(3.33*sigma_ac)

    lr = log_results.get(label, {})
    delta_peak_ac = lr.get('delta_peak_ac', float('nan'))
    sigma_stubbs  = lr.get('sigma_stubbs')
    sigma_tl      = lr.get('sigma_tl')
    sigma_tr      = lr.get('sigma_tr')
    delta_peak_st = lr.get('delta_peak_st')

    make_info_panel(label, sigma_ac, degc_ac, fwhm,
                    delta_peak_ac, snr_lin, snr_db, snr_cr_lin,
                    sigma_stubbs, sigma_tl, sigma_tr, delta_peak_st)
    print(f'  {label}  SNR={snr_db:.1f}dB  σ_AC={sigma_ac:.3f}  '
          f'σ_St={sigma_stubbs:.3f}μs' if sigma_stubbs else f'  {label}')

print(f"\nSaved info_*.png to {PLOTS}/")
