#!/usr/bin/env python3
"""
Full Stubbs analysis + error analysis on all available recordings.
Covers: July 7 (morn+eve), July 8 chirp sweep, July 8 alpha sweep, PPP.
For each waveform computes:
  - sigma(tau_L), sigma(tau_R), sigma(Dtau) measured + propagated
  - xcorr FWHM, xcorr SNR
  - Cramér-Rao lower bound on sigma_tau
  - Whether measured sigma tracks the CR bound
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, correlate
from scipy.io import wavfile
import chirp_stubbs_analysis as st

SDG_SR = 400000; SDG_MAX = 16384
def rr(d, dead=15.0): return SDG_SR/min(int((d+dead)/1000*SDG_SR),SDG_MAX)

OUT = 'stubbs_full_analysis'
os.makedirs(OUT, exist_ok=True)

SR = 384000

def hp(s): return sosfiltfilt(butter(8,30000/(SR/2),btype='high',output='sos'),s)

def measure_xcorr_quality(T_launch, T_return, eval_inc, eval_ref, fs):
    """Measure xcorr FWHM and SNR for both incident and reflected."""
    results = {}
    for name, template, signal in [('launch', T_launch, eval_inc),
                                    ('return', T_return, eval_ref)]:
        xc = correlate(signal, template, mode='full')
        env = np.abs(hilbert(xc)); env /= env.max()+1e-12
        center = len(template)-1
        lags_us = (np.arange(len(xc))-center)/fs*1e6
        # Search ±500μs
        m = (lags_us>=-500)&(lags_us<=500)
        ix = np.argmax(env[m]); roi=env[m]; lag=lags_us[m]
        peak_val = roi[ix]
        # FWHM
        half=peak_val*0.5; li,ri=ix,ix
        while li>0 and roi[li]>half: li-=1
        while ri<len(roi)-1 and roi[ri]>half: ri+=1
        fwhm = lag[ri]-lag[li]
        # SNR: peak / rms of noise region (|lag|>600μs)
        noise_m = np.abs(lags_us)>600
        snr = peak_val/(np.sqrt(np.mean(env[noise_m]**2))+1e-12) if np.any(noise_m) else 0
        results[name] = {'fwhm_us': fwhm, 'snr': snr,
                          'cr_bound_us': fwhm/(3.33*snr) if snr>0 else np.nan}
    return results

def run_one(wav, rep_rate, tag, dead_ms=15.0):
    """Run Stubbs + error analysis on one recording. Returns result dict."""
    try:
        r = st.analyze(wav, rep_rate=rep_rate, output_dir=OUT)
    except Exception as e:
        print(f'  FAILED: {e}')
        return None

    tl = r['tau_launch']; tr = r['tau_return']; dt = r['delta_tau']
    sig_tl = tl.std(); sig_tr = tr.std()
    sig_dt_meas = dt.std()
    sig_dt_prop = np.sqrt(sig_tl**2 + sig_tr**2)   # error propagation

    # Load signal for xcorr quality measurement
    fs2, raw = wavfile.read(wav)
    sig = hp(raw.astype(np.float64)/32768.0)
    slices = st.adaptive_slice_periods(sig, fs2, rep_rate)
    if len(slices) < r['n_template']+5:
        xcorr_q = None
    else:
        # Use last evaluation period for xcorr measurement
        last = sig[slices[-1][0]:slices[-1][1]]
        inc_win = (r['incident_win_ms'][0], r['incident_win_ms'][1])
        ref_win = (r['reflection_win_ms'][0], r['reflection_win_ms'][1])
        eval_inc = st.mask_period(last, fs2, inc_win)
        eval_ref = st.mask_period(last, fs2, ref_win)
        xcorr_q = measure_xcorr_quality(r['T_launch'], r['T_return'],
                                         eval_inc, eval_ref, fs2)

    tl_pk = np.max(np.abs(r['T_launch']))
    tr_pk = np.max(np.abs(r['T_return']))

    cr_launch = xcorr_q['launch']['cr_bound_us'] if xcorr_q else np.nan
    cr_return = xcorr_q['return']['cr_bound_us'] if xcorr_q else np.nan
    fwhm_l    = xcorr_q['launch']['fwhm_us']     if xcorr_q else np.nan
    fwhm_r    = xcorr_q['return']['fwhm_us']     if xcorr_q else np.nan
    snr_l     = xcorr_q['launch']['snr']          if xcorr_q else np.nan
    snr_r     = xcorr_q['return']['snr']          if xcorr_q else np.nan

    return dict(tag=tag, wav=wav,
                n_eval=r['n_eval'], tl_pk=tl_pk, tr_pk=tr_pk,
                sig_tl=sig_tl, sig_tr=sig_tr,
                sig_dt_meas=sig_dt_meas, sig_dt_prop=sig_dt_prop,
                fwhm_l=fwhm_l, fwhm_r=fwhm_r,
                snr_l=snr_l, snr_r=snr_r,
                cr_launch=cr_launch, cr_return=cr_return,
                degC=sig_dt_meas/10.3)

# ── Define all recordings ─────────────────────────────────────────────────────
B7M = 'recordings/20260707_025425_chirp_sweep/raw'
B7E = 'recordings/20260707_214657_chirp_sweep/raw'
B8  = 'recordings/20260708_154521_chirp_sweep/raw'
ALP = 'recordings/20260708_133617_alpha_sweep/raw'
PPP = 'recordings/20260709_173330_flat6kHz_pppB/raw'

RECORDINGS = []

# July 8 chirp sweep (best voltage-calibrated set)
for label, dur, dead in [
    ('bw_20kHz_wide_4ms_comp',      4.0, 15.0),
    ('bw_20kHz_wide_4ms_comp_light',4.0, 15.0),
    ('bw_20kHz_wide_4ms_diff',      4.0, 15.0),
    ('bw_25kHz_wide_2ms_comp',      2.0, 15.0),
    ('bw_6kHz_sym_2ms_flat',        2.0, 15.0),
    ('hf_m15dB_asym_2ms_comp_light',2.0, 15.0),
    ('hf_m15dB_asym_2ms_comp',      2.0, 15.0),
    ('hf_m15dB_asym_2ms_comp_rect', 2.0, 15.0),
    ('hf_m15dB_asym_2ms_diff',      2.0, 15.0),
]:
    RECORDINGS.append((f'{B8}/{label}_received.wav', rr(dur,dead), f'jul8_{label}', dead))

# Alpha sweep (July 8)
for alpha in [0.2,0.3,0.4,0.5,0.6,0.8]:
    for bw_tag,dur in [('sm',2.0),('sm',4.0),('md',2.0),('md',4.0),('lg',2.0),('lg',4.0)]:
        label = f'a{alpha:.1f}_{bw_tag}_{dur:.0f}ms'
        RECORDINGS.append((f'{ALP}/{label}_received.wav', rr(dur,15.0),
                           f'alpha_{label}', 15.0))

# PPP flat 6kHz
RECORDINGS.append((f'{PPP}/bw_6kHz_flat_pppB_received.wav', rr(2.0,15.0),
                   'pppB_flat6kHz', 15.0))

# ── Run all ───────────────────────────────────────────────────────────────────
results = []
n = len(RECORDINGS)
for k, (wav, rep, tag, dead) in enumerate(RECORDINGS):
    if not os.path.exists(wav):
        print(f'[{k+1}/{n}] SKIP (missing): {tag}')
        continue
    print(f'\n[{k+1}/{n}] {tag}')
    r = run_one(wav, rep, tag, dead)
    if r: results.append(r)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*130}')
print(f'{"Tag":<35} {"σ(τL)":>8} {"σ(τR)":>8} {"σ(Δτ)_meas":>11} '
      f'{"σ(Δτ)_prop":>11} {"FWHM_R":>8} {"SNR_R":>7} {"CR_R":>8} '
      f'{"meas/CR":>8} {"°C/ping":>9}')
print('-'*130)
results.sort(key=lambda r: r['sig_dt_meas'])
for r in results:
    meas_cr = r['sig_dt_meas']/r['cr_return'] if r['cr_return']>0 else np.nan
    print(f"  {r['tag']:<33} {r['sig_tl']:>8.3f} {r['sig_tr']:>8.3f} "
          f"{r['sig_dt_meas']:>11.3f} {r['sig_dt_prop']:>11.3f} "
          f"{r['fwhm_r']:>8.1f} {r['snr_r']:>7.1f} {r['cr_return']:>8.3f} "
          f"{meas_cr:>8.2f} {r['degC']:>9.4f}")

# ── Error propagation plot ────────────────────────────────────────────────────
valid = [r for r in results if not np.isnan(r['cr_return']) and r['sig_dt_meas']<20]
if valid:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1: Measured vs propagated sigma(Dtau)
    ax = axes[0]
    meas = [r['sig_dt_meas'] for r in valid]
    prop = [r['sig_dt_prop'] for r in valid]
    ax.scatter(prop, meas, s=30, alpha=0.7)
    lim = max(max(meas), max(prop))*1.1
    ax.plot([0,lim],[0,lim],'k--',lw=1,label='y=x (perfect propagation)')
    ax.set_xlabel('σ(Δτ) propagated = √(σL²+σR²)  (μs)')
    ax.set_ylabel('σ(Δτ) measured  (μs)')
    ax.set_title('Error propagation check')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 2: Measured sigma(tau_R) vs CR bound
    ax = axes[1]
    cr = [r['cr_return'] for r in valid]
    sig_r = [r['sig_tr'] for r in valid]
    ax.scatter(cr, sig_r, s=30, alpha=0.7, c='C1')
    lim = max(max(cr),max(sig_r))*1.1
    ax.plot([0,lim],[0,lim],'k--',lw=1,label='y=x (at CR bound)')
    ax.set_xlabel('Cramér-Rao bound = FWHM/(3.33×SNR)  (μs)')
    ax.set_ylabel('σ(τ_return) measured  (μs)')
    ax.set_title('σ(τR) vs Cramér-Rao bound')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 3: meas/CR ratio vs FWHM (bandwidth proxy)
    ax = axes[2]
    fwhm_r = [r['fwhm_r'] for r in valid]
    ratio = [r['sig_dt_meas']/r['cr_return'] for r in valid]
    ax.scatter(fwhm_r, ratio, s=30, alpha=0.7, c='C2')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='at CR bound')
    ax.axhline(3.0, color='gray', ls=':', lw=1, label='3× CR (env. dominated)')
    ax.set_xlabel('Reflected xcorr FWHM (μs)')
    ax.set_ylabel('σ(Δτ)_meas / CR_bound')
    ax.set_title('Excess noise above CR bound\n(→1 = instrument-limited, →high = env-limited)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    ax.set_ylim(0, min(20, max(ratio)*1.2))

    plt.suptitle('BATPING error analysis — all July 7-9 recordings', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/error_analysis.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved: {OUT}/error_analysis.png')
