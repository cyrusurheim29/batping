#!/usr/bin/env python3
"""Combine per-segment oneway_sync_analysis NPZs into a single 30-min view.

Given the folder pattern:
    <recording_dir>/oneway_analysis/<wav_stem>/per_ping.npz

concatenates all segments' per-ping arrays in file-order, offsetting t_pulse_s
by (seg_index × segment_duration) so the time axis is continuous.

Also aligns absolute TOF baselines so segments join without a discontinuity —
though we save both raw-baseline and aligned-baseline versions for inspection.

Usage:
    python3 oneway_combine_segments.py <recording_dir>
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOF_SENS_US_PER_DEG = 23.2

if len(sys.argv) < 2:
    sys.exit("usage: oneway_combine_segments.py <recording_dir> "
             "[--subdir oneway_delayedchirp_analysis] [--stitch-xcorr] "
             "[--rep-rate 29.41] [--search-ms 18 28] [--ch-a-fs-v 2] [--ch-b-fs-v 5]")
REC_DIR = sys.argv[1]
SUBDIR  = 'oneway_analysis'
STITCH_XCORR = '--stitch-xcorr' in sys.argv
if '--subdir' in sys.argv:
    SUBDIR = sys.argv[sys.argv.index('--subdir') + 1]
# optional template-rebuild params (only used if --stitch-xcorr)
STITCH_REP_RATE  = float(sys.argv[sys.argv.index('--rep-rate')  + 1]) if '--rep-rate'  in sys.argv else 29.41
STITCH_SEARCH_MS = (float(sys.argv[sys.argv.index('--search-ms') + 1]),
                    float(sys.argv[sys.argv.index('--search-ms') + 2])) if '--search-ms' in sys.argv else (18.0, 28.0)
STITCH_CH_A_FS_V = float(sys.argv[sys.argv.index('--ch-a-fs-v') + 1]) if '--ch-a-fs-v' in sys.argv else 2.0
STITCH_CH_B_FS_V = float(sys.argv[sys.argv.index('--ch-b-fs-v') + 1]) if '--ch-b-fs-v' in sys.argv else 5.0

# Find all per-segment analyses (sorted by seg number) — directories only,
# so sibling PNGs like diff_temp_timeseries_seg01.png don't get swept in.
seg_dirs = sorted(d for d in glob.glob(os.path.join(REC_DIR, SUBDIR, '*_seg*'))
                  if os.path.isdir(d))
print(f"Found {len(seg_dirs)} per-segment analyses:")
for d in seg_dirs:
    print(f"  {os.path.basename(d)}")
if len(seg_dirs) < 1:
    sys.exit("No per-segment analyses found. Run oneway_sync_analysis.py on each segment first.")

# ── Optional: between-segment stitching via boundary-median alignment ────
#
# Each segment is analyzed with its OWN `cha_offset_samp` (the Phase 1 median
# offset of Ch A arrival relative to Ch B pulse, built from that segment's
# first 5 s).  When the acoustic path drifts between segments, each segment's
# cha_offset_samp picks a different integer-sample reference, so TOFs across
# segments live in different reference frames → visible step jumps.
#
# Since the WAV capture is physically continuous (streaming mode), the TRUE
# TOF at the end of seg N must equal the TRUE TOF at the start of seg N+1.
# Any observed jump at the boundary IS the artifact.  Correction:
#   jump_{N,N+1} = median(first N0 inlier TOFs of seg N+1)
#                - median(last  N0 inlier TOFs of seg N)
#   apply cumulative sum of jumps to each seg's TOFs so they live in seg01's
#   reference frame.
#
# [Prior "xcorr-template" implementation returned ≈0 lag because both
#  templates were envelope-centered — the jump isn't in the template
#  waveform, it's in the integer-sample reference choice.]
stitch_offsets_us_para  = [0.0] * len(seg_dirs)
stitch_offsets_us_shape = [0.0] * len(seg_dirs)
N0_BOUNDARY = 200
if STITCH_XCORR:
    # First pass: load segments, extract boundary medians
    print(f"\n── Boundary-median stitching (robust TOF alignment, N0={N0_BOUNDARY}) ──")
    seg_tofs = []
    for seg_dir in seg_dirs:
        d = np.load(os.path.join(seg_dir, 'per_ping.npz'))
        seg_tofs.append(dict(
            tof_p=d['tof_para_us'],  tof_s=d['tof_shape_us'],
            inl_p=d['inl_para'],     inl_s=d['inl_shape'],
        ))

    def _med_tail(arr, inl, n):
        v = arr[inl]
        if len(v) < n: return float(np.median(v))
        return float(np.median(v[-n:]))

    def _med_head(arr, inl, n):
        v = arr[inl]
        if len(v) < n: return float(np.median(v))
        return float(np.median(v[:n]))

    for k in range(1, len(seg_tofs)):
        p_prev = seg_tofs[k-1];  p_curr = seg_tofs[k]
        jump_p = _med_head(p_curr['tof_p'], p_curr['inl_p'], N0_BOUNDARY) \
               - _med_tail(p_prev['tof_p'], p_prev['inl_p'], N0_BOUNDARY)
        jump_s = _med_head(p_curr['tof_s'], p_curr['inl_s'], N0_BOUNDARY) \
               - _med_tail(p_prev['tof_s'], p_prev['inl_s'], N0_BOUNDARY)
        stitch_offsets_us_para [k] = stitch_offsets_us_para [k-1] + jump_p
        stitch_offsets_us_shape[k] = stitch_offsets_us_shape[k-1] + jump_s
        print(f"  seg{k:02d}→seg{k+1:02d}:  Δτ_jump para {jump_p:+9.3f} μs  shape {jump_s:+9.3f} μs   "
              f"cumulative para {stitch_offsets_us_para[k]:+9.3f} μs  shape {stitch_offsets_us_shape[k]:+9.3f} μs")

# ── Load per-segment NPZs, offset time + (optionally) stitch ─────────────
all_t, all_tof_para, all_tof_shape = [], [], []
all_sig_para, all_sig_shape = [], []
all_sig_B_para, all_sig_B_shape = [], []
all_sig_diff_para, all_sig_diff_shape = [], []
all_inl_para, all_inl_shape = [], []
t_offset = 0.0
rep_rate = None

print()
for k, seg_dir in enumerate(seg_dirs):
    d = np.load(os.path.join(seg_dir, 'per_ping.npz'))
    t_seg = d['t_pulse_s']
    all_t.append(t_seg + t_offset)
    stitch_p = stitch_offsets_us_para [k]
    stitch_s = stitch_offsets_us_shape[k]
    all_tof_para .append(d['tof_para_us']  - stitch_p)
    all_tof_shape.append(d['tof_shape_us'] - stitch_s)
    all_sig_para .append(d['sigma_para_us'])
    all_sig_shape.append(d['sigma_shape_us'])
    # Ch B + diff sigmas (added for shape-only plots) — fallback to zeros
    # if old-format NPZ doesn't have them
    all_sig_B_para   .append(d['sigma_B_para_us']    if 'sigma_B_para_us'    in d.files else np.zeros_like(d['sigma_para_us']))
    all_sig_B_shape  .append(d['sigma_B_shape_us']   if 'sigma_B_shape_us'   in d.files else np.zeros_like(d['sigma_shape_us']))
    all_sig_diff_para .append(d['sigma_diff_para_us']  if 'sigma_diff_para_us'  in d.files else np.zeros_like(d['sigma_para_us']))
    all_sig_diff_shape.append(d['sigma_diff_shape_us'] if 'sigma_diff_shape_us' in d.files else np.zeros_like(d['sigma_shape_us']))
    all_inl_para .append(d['inl_para'])
    all_inl_shape.append(d['inl_shape'])
    if rep_rate is None:
        rep_rate = float(d['rep_rate'])
    # Advance t_offset by this segment's audio duration
    seg_duration = t_seg[-1] - t_seg[0] + 1.0 / rep_rate
    t_offset += seg_duration
    print(f"  {os.path.basename(seg_dir):55s}  N={len(t_seg):>6,}  "
          f"span={t_seg[0]/60:.2f}-{t_seg[-1]/60:.2f} min  baseline_TOF={float(np.mean(d['tof_para_us'][:100])):.3f} μs  "
          f"stitch_para={stitch_p:+9.3f} μs  stitch_shape={stitch_s:+9.3f} μs")

t         = np.concatenate(all_t)
tof_para  = np.concatenate(all_tof_para)
tof_shape = np.concatenate(all_tof_shape)
sig_para  = np.concatenate(all_sig_para)
sig_shape = np.concatenate(all_sig_shape)
sig_B_para    = np.concatenate(all_sig_B_para)
sig_B_shape   = np.concatenate(all_sig_B_shape)
sig_diff_para  = np.concatenate(all_sig_diff_para)
sig_diff_shape = np.concatenate(all_sig_diff_shape)
inl_para  = np.concatenate(all_inl_para)
inl_shape = np.concatenate(all_inl_shape)

# Δτ vs first-100-pings baseline
baseline_para  = float(np.mean(tof_para[:100]))
baseline_shape = float(np.mean(tof_shape[:100]))
dtau_para  = tof_para  - baseline_para
dtau_shape = tof_shape - baseline_shape

# HF-detrended σ
win_1s = max(3, int(rep_rate) | 1)
r_para  = pd.Series(np.where(inl_para,  dtau_para,  np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values
r_shape = pd.Series(np.where(inl_shape, dtau_shape, np.nan)).rolling(win_1s, center=True, min_periods=win_1s//2).mean().values

hf_para  = (dtau_para  - r_para )[inl_para  & np.isfinite(r_para )]
hf_shape = (dtau_shape - r_shape)[inl_shape & np.isfinite(r_shape)]
sig_hf_para  = float(np.std(hf_para))
sig_hf_shape = float(np.std(hf_shape))

sig_full_para  = float(np.std(dtau_para [inl_para ]))
sig_full_shape = float(np.std(dtau_shape[inl_shape]))

print(f"\n═══ COMBINED 30-min SUMMARY ═══")
print(f"  N pings total     : {len(t):,}")
print(f"  duration          : {(t[-1]-t[0])/60:.2f} min")
print(f"  baseline TOF      : para {baseline_para:.3f} μs   shape {baseline_shape:.3f} μs")
print(f"  implied L         : {baseline_para/1e6*343*100:.1f} cm  (mic-to-speaker path @ 343 m/s)")
print(f"  net TOF drift     : para {tof_para[-1]-tof_para[0]:+.3f} μs   ({(tof_para[-1]-tof_para[0])/TOF_SENS_US_PER_DEG*1000:+.0f} mK apparent ΔT)")
print(f"  σ(Δτ) full-run (thermal drift INCLUDED — mostly the room warming up):")
print(f"    para  {sig_full_para:7.3f} μs  ({sig_full_para/TOF_SENS_US_PER_DEG*1000:6.1f} mK)")
print(f"    shape {sig_full_shape:7.3f} μs  ({sig_full_shape/TOF_SENS_US_PER_DEG*1000:6.1f} mK)")
print(f"  σ(Δτ) HF-only (1-s detrended → per-ping noise floor):")
print(f"    para  {sig_hf_para:7.4f} μs  ({sig_hf_para/TOF_SENS_US_PER_DEG*1000:6.1f} mK/ping)  ★")
print(f"    shape {sig_hf_shape:7.4f} μs  ({sig_hf_shape/TOF_SENS_US_PER_DEG*1000:6.1f} mK/ping)")

# ── Plot: 4-panel combined figure ─────────────────────────────────────────
t_min = t / 60.0
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

# (1) Raw TOF over full 30 min — the thermal drift signature
ax = axes[0, 0]
ax.plot(t_min[inl_para],  tof_para[inl_para],  '.', ms=0.8, alpha=0.5, color='#1f77b4', label='parabolic')
ax.plot(t_min[inl_shape], tof_shape[inl_shape], '.', ms=0.8, alpha=0.5, color='#e6444d', label='shape-fit')
# Also plot rolling-mean overlay so the drift is clearly separable from noise
win_60s = max(3, int(60 * rep_rate) | 1)
r_para_60s = pd.Series(np.where(inl_para, tof_para, np.nan)).rolling(win_60s, center=True, min_periods=win_60s//3).mean().values
ax.plot(t_min, r_para_60s, '-', color='k', lw=1.5, alpha=0.8, label='60-s rolling mean')
# Segment boundary markers
seg_dur = t_min[-1] / len(seg_dirs)
for i in range(1, len(seg_dirs)):
    ax.axvline(seg_dur * i, color='gray', ls=':', lw=0.6, alpha=0.6)
ax.set_xlabel('Time (min)'); ax.set_ylabel('TOF (μs)')
ax.set_title(f'Raw TOF over full 30 min   net drift: {tof_para[-1]-tof_para[0]:+.2f} μs ({(tof_para[-1]-tof_para[0])/TOF_SENS_US_PER_DEG*1000:+.0f} mK)')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

# (2) HF residual — drift removed
ax = axes[0, 1]
ax.plot(t_min[inl_para & np.isfinite(r_para)],  (dtau_para  - r_para )[inl_para  & np.isfinite(r_para )], '.', ms=0.8, alpha=0.4, color='#1f77b4', label=f'parabolic  σ_HF={sig_hf_para:.3f} μs')
ax.plot(t_min[inl_shape & np.isfinite(r_shape)], (dtau_shape - r_shape)[inl_shape & np.isfinite(r_shape)], '.', ms=0.8, alpha=0.4, color='#e6444d', label=f'shape-fit  σ_HF={sig_hf_shape:.3f} μs')
ax.axhline(0, color='k', lw=0.4)
for i in range(1, len(seg_dirs)):
    ax.axvline(seg_dur * i, color='gray', ls=':', lw=0.6, alpha=0.6)
ax.set_xlabel('Time (min)'); ax.set_ylabel('Δτ − rolling_mean (μs)')
ax.set_title('HF residual (thermal drift removed via 1-s rolling mean)')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

# (3) Histogram of HF residual
ax = axes[1, 0]
lo = min(np.percentile(hf_para, 1), np.percentile(hf_shape, 1))
hi = max(np.percentile(hf_para, 99), np.percentile(hf_shape, 99))
bins = np.linspace(lo, hi, 100)
ax.hist(hf_para,  bins=bins, alpha=0.55, color='#1f77b4',
         label=f'parabolic σ_HF={sig_hf_para:.3f} μs ({sig_hf_para/TOF_SENS_US_PER_DEG*1000:.1f} mK)')
ax.hist(hf_shape, bins=bins, alpha=0.55, color='#e6444d',
         label=f'shape-fit σ_HF={sig_hf_shape:.3f} μs ({sig_hf_shape/TOF_SENS_US_PER_DEG*1000:.1f} mK)')
ax.set_xlabel('HF residual (μs)'); ax.set_ylabel('Count')
ax.set_title(f'Per-ping noise floor histogram   (N={len(hf_para):,} pings, drift-subtracted)')
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# (4) Rolling-1-min σ over full 30 min
ax = axes[1, 1]
win_1min = max(3, int(60 * rep_rate) | 1)
r_std_para  = pd.Series(np.where(inl_para,  dtau_para,  np.nan)).rolling(win_1min, center=True, min_periods=win_1min//3).std().values
r_std_shape = pd.Series(np.where(inl_shape, dtau_shape, np.nan)).rolling(win_1min, center=True, min_periods=win_1min//3).std().values
ax.plot(t_min, r_std_para,  '-', color='#1f77b4', lw=1.0, label='parabolic')
ax.plot(t_min, r_std_shape, '-', color='#e6444d', lw=1.0, label='shape-fit')
for i in range(1, len(seg_dirs)):
    ax.axvline(seg_dur * i, color='gray', ls=':', lw=0.6, alpha=0.6)
ax.set_xlabel('Time (min)'); ax.set_ylabel('Rolling-1-min σ(Δτ) (μs)')
ax.set_title('σ(Δτ) stability over 30-min recording')
ax.grid(alpha=0.3); ax.legend(fontsize=9)

fig.suptitle(f'One-way BATPING 4.5m — combined {(t[-1]-t[0])/60:.1f}-min analysis  '
              f'(σ_HF: parabolic {sig_hf_para/TOF_SENS_US_PER_DEG*1000:.1f} mK/ping  |  '
              f'shape-fit {sig_hf_shape/TOF_SENS_US_PER_DEG*1000:.1f} mK/ping)',
              fontsize=12, fontweight='bold')
fig.tight_layout()

out_dir = os.path.join(REC_DIR, SUBDIR)
out_png = os.path.join(out_dir, 'tof_combined_30min.png')
fig.savefig(out_png, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nwrote {out_png}")

# Save combined NPZ
out_npz = os.path.join(out_dir, 'per_ping_combined.npz')
np.savez(out_npz,
          t_pulse_s=t, tof_para_us=tof_para, tof_shape_us=tof_shape,
          sigma_para_us=sig_para, sigma_shape_us=sig_shape,
          sigma_B_para_us=sig_B_para, sigma_B_shape_us=sig_B_shape,
          sigma_diff_para_us=sig_diff_para, sigma_diff_shape_us=sig_diff_shape,
          inl_para=inl_para, inl_shape=inl_shape,
          dtau_para_us=dtau_para, dtau_shape_us=dtau_shape,
          sig_hf_para=sig_hf_para, sig_hf_shape=sig_hf_shape,
          baseline_para_us=baseline_para, baseline_shape_us=baseline_shape,
          rep_rate=rep_rate, tof_sens_us_per_deg=TOF_SENS_US_PER_DEG)
print(f"wrote {out_npz}")
