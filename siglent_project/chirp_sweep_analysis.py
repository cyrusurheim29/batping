#!/usr/bin/env python3
"""
Full analysis of the July 10 chirp sweep (voltage-calibrated).
For each waveform:
  - Stubbs 30s with adaptive slicer
  - Autocorr 30s with adaptive slicer
  - Standalone info panel PNG
  - Per-waveform subdir with all 3 plots
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, correlate
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont
import chirp_stubbs_analysis as st

REC  = 'recordings/20260710_030748_chirp_sweep'
BASE = f'{REC}/raw'
PLOTS= f'{REC}/plots'
SDIR = f'{REC}/stubbs'
os.makedirs(SDIR, exist_ok=True)

SR = 384000
SDG_SR=400000; SDG_MAX=16384
def rr(d, dead=15.0): return SDG_SR/min(int((d+dead)/1000*SDG_SR),SDG_MAX)
def hp(s): return sosfiltfilt(butter(8,30000/(SR/2),btype='high',output='sos'),s)

CONFIGS = [
    ('bw_20kHz_wide_4ms_comp',      4.0, 15.0),
    ('bw_20kHz_wide_4ms_comp_light',4.0, 15.0),
    ('bw_20kHz_wide_4ms_diff',      4.0, 15.0),
    ('bw_25kHz_wide_2ms_comp',      2.0, 15.0),
    ('bw_6kHz_sym_2ms_flat',        2.0, 23.0),
    ('hf_m15dB_asym_2ms_comp_light',2.0, 15.0),
    ('hf_m15dB_asym_2ms_comp',      2.0, 15.0),
    ('hf_m15dB_asym_2ms_comp_rect', 2.0, 15.0),
    ('hf_m15dB_asym_2ms_diff',      2.0, 15.0),
]

def sigma_tof_adaptive(sig, rep_hz, tof_min=4000, tof_max=8500):
    """Per-period sigma_TOF using adaptive slicer — correct at 30s."""
    slices = st.adaptive_slice_periods(sig, SR, rep_hz)
    tofs, dsigs = [], []
    for s_,e_ in slices:
        seg = sig[s_:e_]
        ac = correlate(seg, seg, mode='full')
        ctr = len(ac)//2
        lu = (np.arange(len(ac))-ctr)/SR*1e6
        ac /= np.max(np.abs(ac))
        ev = np.abs(hilbert(ac)); ev /= ev.max()
        m = (lu>=tof_min)&(lu<=tof_max)
        if not np.any(m): continue
        ix = np.argmax(ev[m]); r=ev[m]; lg=lu[m]
        # 11-pt parabolic fit
        hw=5; lo=max(0,ix-hw); hi=min(len(r),ix+hw+1)
        x=np.arange(lo,hi,dtype=float)-ix; y=r[lo:hi]
        if len(y)>=5 and np.ptp(y)>1e-12:
            coeffs,res,_,_,_=np.polyfit(x,y,2,full=True)
            a,b=coeffs[0],coeffs[1]
            if abs(a)>1e-10:
                dl=float(np.clip(-b/(2*a),-hw,hw))
                tofs.append(float(lg[ix])+dl*(lg[1]-lg[0]))
                rms=float(np.sqrt(res[0]/len(y))) if len(res) else 0.1
                step=float(lg[1]-lg[0])
                dsigs.append(rms/(abs(a)*float(np.sqrt(np.sum(x**2))))*step)
                continue
        tofs.append(float(lg[ix]))
        dsigs.append(float(lg[1]-lg[0]))
    if len(tofs)<5: return None, None, 0
    return float(np.std(tofs)), float(np.mean(dsigs)) if dsigs else None, len(tofs)

# ── Run analysis ──────────────────────────────────────────────────────────────
try:
    font_b = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    font   = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    font_v = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    font_s = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
except:
    font_b = font = font_v = font_s = ImageFont.load_default()

results = []
for label, dur_ms, dead_ms in CONFIGS:
    wav = f'{BASE}/{label}_received.wav'
    if not os.path.exists(wav): print(f'SKIP {label}'); continue
    print(f'\n{"="*55}\n  {label}\n{"="*55}')

    fs2, raw = wavfile.read(wav)
    sig = hp(raw.astype(np.float64)/32768.0)
    rep = rr(dur_ms, dead_ms)

    # 1. Autocorr 30s adaptive
    sigma_ac, dpeak_ac, n_ac = sigma_tof_adaptive(sig, rep)
    fwhm_ac = None
    if sigma_ac:
        # Estimate FWHM from one period autocorr
        slices_tmp = st.adaptive_slice_periods(sig, SR, rep)
        if slices_tmp:
            i0 = min(10, len(slices_tmp)-1)
            seg = sig[slices_tmp[i0][0]:slices_tmp[i0][1]]
            ac = correlate(seg,seg,mode='full'); ctr=len(ac)//2
            lu=(np.arange(len(ac))-ctr)/SR*1e6; ac/=np.max(np.abs(ac))
            ev=np.abs(hilbert(ac)); ev/=ev.max()
            m=(lu>=4000)&(lu<=8500)
            if np.any(m):
                ix=np.argmax(ev[m]); r=ev[m]; lg=lu[m]
                half=r[ix]*0.5; li,ri=ix,ix
                while li>0 and r[li]>half: li-=1
                while ri<len(r)-1 and r[ri]>half: ri+=1
                fwhm_ac = lg[ri]-lg[li]
        snr_cr = (fwhm_ac/(3.33*sigma_ac)) if fwhm_ac and sigma_ac else None
    print(f'  Autocorr: sigma={sigma_ac:.3f}us  fwhm={fwhm_ac:.0f}us  dpeak={dpeak_ac*1000:.0f}ns  N={n_ac}' if sigma_ac else f'  Autocorr: FAILED')

    # 2. Stubbs 30s
    try:
        r_st = st.analyze(wav, rep_rate=rep, output_dir=SDIR)
        sigma_st     = r_st['std_delta_tau_us']
        sigma_tl     = r_st['tau_launch'].std()
        sigma_tr     = r_st['tau_return'].std()
        dpeak_st     = r_st.get('mean_fit_sigma_us')
        n_eval       = r_st['n_eval']
        tl_pk        = np.max(np.abs(r_st['T_launch']))
        tr_pk        = np.max(np.abs(r_st['T_return']))
    except Exception as e:
        print(f'  Stubbs FAILED: {e}')
        sigma_st=sigma_tl=sigma_tr=dpeak_st=None; n_eval=0; tl_pk=tr_pk=0
    print(f'  Stubbs: sigma={sigma_st:.3f}us  sigma_tL={sigma_tl:.3f}  sigma_tR={sigma_tr:.3f}' if sigma_st else '  Stubbs: FAILED')

    results.append(dict(label=label, dur_ms=dur_ms, dead_ms=dead_ms,
                        sigma_ac=sigma_ac, dpeak_ac=dpeak_ac, fwhm_ac=fwhm_ac,
                        snr_cr=snr_cr, n_ac=n_ac,
                        sigma_st=sigma_st, sigma_tl=sigma_tl, sigma_tr=sigma_tr,
                        dpeak_st=dpeak_st, n_eval=n_eval, tl_pk=tl_pk, tr_pk=tr_pk))

    # 3. Info panel
    fig = plt.figure(figsize=(4.2, 6.0), facecolor='#f8f8f8')
    ax = fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_facecolor('#f8f8f8')
    y=0.97; lh=0.060; lhs=0.047

    def txt(x,y_,s,**kw):
        ax.text(x,y_,s,transform=ax.transAxes,va='top',
                fontsize=kw.pop('fs',9.5),**kw)

    txt(0.5,y,label,fs=10.5,fontweight='bold',ha='center',color='#1a1a2e'); y-=lh*1.0
    txt(0.5,y,f'{dur_ms:.0f}ms  dead={dead_ms:.0f}ms  {n_ac} periods (30s)',
        fs=8,ha='center',color='#666'); y-=lh*1.3
    ax.axhline(y,color='#aaa',lw=0.8,xmin=0.05,xmax=0.95); y-=lh*0.4

    # Autocorr section
    txt(0.05,y,'Per-period autocorr  (30s, adaptive)',fs=8.5,
        color='#2c5f8a',fontweight='bold'); y-=lh*0.9
    for key,val,red in [
        ('σ_TOF',    f'{sigma_ac:.3f} μs' if sigma_ac else 'N/A', True),
        ('°C/ping',  f'{sigma_ac/10.3:.4f}' if sigma_ac else 'N/A', True),
        ('FWHM',     f'{fwhm_ac:.0f} μs' if fwhm_ac else 'N/A', False),
        ('δ_peak',   f'{dpeak_ac*1000:.0f} ns' if dpeak_ac else 'N/A', False),
        ('SNR (CR)', f'{snr_cr:.0f}×  ({20*np.log10(snr_cr):.1f} dB)' if snr_cr else 'N/A', False),
    ]:
        txt(0.08,y,f'{key}:',fs=8,color='#666'); y-=lhs
        txt(0.12,y,val,fs=9,color='#c0392b' if red else '#333',
            fontweight='bold' if red else 'normal'); y-=lh*0.8

    y-=lh*0.3; ax.axhline(y,color='#ccc',lw=0.6,xmin=0.05,xmax=0.95); y-=lh*0.4

    # Stubbs section
    txt(0.05,y,'Stubbs matched filter  (30s)',fs=8.5,
        color='#27612e',fontweight='bold'); y-=lh*0.9
    for key,val,red in [
        ('σ(Δτ)',   f'{sigma_st:.3f} μs' if sigma_st else 'N/A', True),
        ('°C/ping', f'{sigma_st/10.3:.4f}' if sigma_st else 'N/A', True),
        ('σ(τ_L)',  f'{sigma_tl:.3f} μs' if sigma_tl else 'N/A', False),
        ('σ(τ_R)',  f'{sigma_tr:.3f} μs' if sigma_tr else 'N/A', False),
        ('δ_peak',  f'{dpeak_st*1000:.0f} ns' if dpeak_st else 'N/A', False),
        ('T_L pk',  f'{tl_pk:.3f}' if tl_pk else 'N/A', False),
        ('T_R pk',  f'{tr_pk:.3f}' if tr_pk else 'N/A', False),
    ]:
        txt(0.08,y,f'{key}:',fs=8,color='#666'); y-=lhs
        txt(0.12,y,val,fs=9,color='#c0392b' if red else '#333',
            fontweight='bold' if red else 'normal'); y-=lh*0.75

    y-=lh*0.2; ax.axhline(y,color='#ccc',lw=0.6,xmin=0.05,xmax=0.95); y-=lh*0.4
    if sigma_ac and sigma_st and sigma_st < 20:
        ratio = sigma_ac/sigma_st
        if ratio > 1.1:
            comp,col = f'Stubbs {ratio:.1f}× better','#27612e'
        elif ratio < 0.9:
            comp,col = f'Autocorr {1/ratio:.1f}× better','#2c5f8a'
        else:
            comp,col = 'Methods comparable','#666'
        txt(0.5,y,comp,fs=9,ha='center',color=col,fontweight='bold')

    subdir = os.path.join(PLOTS, label)
    os.makedirs(subdir, exist_ok=True)
    plt.savefig(os.path.join(subdir,f'info_{label}.png'),
                dpi=150,bbox_inches='tight',facecolor='#f8f8f8')
    plt.close()

    # Copy recording plot into subdir
    import shutil
    src = os.path.join(PLOTS,f'{label}.png')
    dst = os.path.join(subdir,f'{label}.png')
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src,dst)
    # Move Stubbs plot
    stubbs_src = os.path.join(SDIR,f'stubbs_{label}_received.png')
    stubbs_dst = os.path.join(subdir,f'stubbs_{label}_received.png')
    if os.path.exists(stubbs_src) and not os.path.exists(stubbs_dst):
        shutil.move(stubbs_src, stubbs_dst)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(f'{"Label":<35} {"AC_30s":>8} {"St_30s":>8} {"Winner":>18} {"AC°C":>8} {"St°C":>8}')
print('-'*80)
stubbs_better=0; ac_better=0; equal=0
for r in results:
    ac=r['sigma_ac']; st_=r['sigma_st']
    if not ac: ac_str='FAIL'
    else: ac_str=f'{ac:.3f}'
    if not st_ or st_>20: st_str='FAIL'
    else: st_str=f'{st_:.3f}'

    if ac and st_ and st_<20:
        ratio=ac/st_
        if ratio>1.1: winner=f'Stubbs {ratio:.1f}×'; stubbs_better+=1
        elif ratio<0.9: winner=f'AC {1/ratio:.1f}×'; ac_better+=1
        else: winner='~equal'; equal+=1
    elif not ac: winner='Stubbs (AC fail)'
    else: winner='AC (St fail)'

    ac_c=f'{ac/10.3:.4f}' if ac else 'N/A'
    st_c=f'{st_/10.3:.4f}' if st_ and st_<20 else 'N/A'
    print(f'  {r["label"]:<33} {ac_str:>8} {st_str:>8} {winner:>18} {ac_c:>8} {st_c:>8}')

print(f'\nStubbs better: {stubbs_better}  AC better: {ac_better}  Equal: {equal}')
