#!/usr/bin/env python3
"""Regenerate 30-s stubbs-LS-fit style per-chunk 3-panel plots from existing
WAV + per_ping.npz files, WITHOUT re-running the full matched-filter pipeline.

For each segment WAV:
  1. rebuild Ch A + Ch B templates from the first ~5 s (same as pipeline)
  2. for each 30-s chunk:
       - pick the ping closest to the chunk midpoint
       - compute CCFs (Ch A and Ch B) for THAT single ping
       - call plot_per_chunk() with CCFs + per-chunk TOF slice from NPZ

Usage:
    python3 oneway_delayedchirp_chunk_plots.py <recording_dir> \\
        [--every-min N]        # sample every N minutes (default 1 = every 30-s chunk)
        [--rep-rate 29.41]
        [--search-ms 18 28]
        [--ch-a-fs-v 2] [--ch-b-fs-v 5]
"""
import os, sys, glob, argparse
import numpy as np
import soundfile as sf
from scipy.signal import correlate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oneway_delayedchirp_analysis import (
    hp_filter, find_chb_peaks_global, find_cha_initial_arrivals,
    build_template, analyze_ping_pair, plot_per_chunk,
    N_TEMPLATE, ARRIVAL_WIN_HW_MS, CHUNK_S, F_LO, F_HI, CHIRP_RATE_HZ_S,
    SOUND_SPEED_M_S, DC_DT_M_S_PER_C,
)

ap = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
ap.add_argument('rec_dir')
ap.add_argument('--every-min', type=float, default=0.5,
                help='sample every N minutes (default 0.5 → every 30-s chunk)')
ap.add_argument('--rep-rate',  type=float, default=29.41)
ap.add_argument('--search-ms', type=float, nargs=2, default=[18.0, 28.0])
ap.add_argument('--ch-a-fs-v', type=float, default=2.0)
ap.add_argument('--ch-b-fs-v', type=float, default=5.0)
args = ap.parse_args()

REC    = args.rec_dir
SUBDIR = 'oneway_delayedchirp_analysis'

# ── Discover segment WAVs + NPZs ──────────────────────────────────────────
wavs = sorted(glob.glob(os.path.join(REC, '*_seg??.wav')))
wavs = [w for w in wavs if os.path.getsize(w) > 1_000_000]
if not wavs:
    wavs = sorted(glob.glob(os.path.join(REC, '*_received.wav')))
if not wavs:
    sys.exit(f"no WAVs found in {REC}")

print(f"Found {len(wavs)} WAV segment(s)")
for w in wavs:
    print(f"  {os.path.basename(w)}")

# ── Process each segment ──────────────────────────────────────────────────
for wav in wavs:
    wav_stem = os.path.splitext(os.path.basename(wav))[0]
    seg_dir  = os.path.join(REC, SUBDIR, wav_stem)
    npz_path = os.path.join(seg_dir, 'per_ping.npz')
    if not os.path.exists(npz_path):
        print(f"  SKIP (no NPZ): {wav_stem}")
        continue
    out_chunk_dir = os.path.join(seg_dir, 'chunk_plots')
    os.makedirs(out_chunk_dir, exist_ok=True)

    print(f"\n──── {wav_stem} ────")
    d = np.load(npz_path)
    t_pulse   = d['t_pulse_s']
    tof_para  = d['tof_para_us']
    tof_shape = d['tof_shape_us']
    sig_para  = d['sigma_para_us']
    sig_shape = d['sigma_shape_us']
    inl_para  = d['inl_para']
    inl_shape = d['inl_shape']
    tof_sens  = float(d['tof_sens_us_per_deg']) if 'tof_sens_us_per_deg' in d.files else 48.0
    base_para  = float(np.mean(tof_para [inl_para ][:100]))
    base_shape = float(np.mean(tof_shape[inl_shape][:100]))
    dtau_para  = tof_para  - base_para
    dtau_shape = tof_shape - base_shape

    # Rebuild templates from first ~5 s (same seconds-needed logic as pipeline)
    info = sf.info(wav)
    fs = info.samplerate
    seconds_needed = max(5.0, (N_TEMPLATE + 20) / args.rep_rate + 1.0)
    n_read = int(seconds_needed * fs)
    with sf.SoundFile(wav, 'r') as f:
        hdr = f.read(n_read, dtype='int16')
    ch_a_hp_tpl = hp_filter(hdr[:, 0].astype(np.float32) / 32768 * args.ch_a_fs_v, fs)
    ch_b_hp_tpl = hp_filter(hdr[:, 1].astype(np.float32) / 32768 * args.ch_b_fs_v, fs)
    chb_peaks_tpl = find_chb_peaks_global(ch_b_hp_tpl, fs, args.rep_rate)
    ch_b_template, win_samples = build_template(ch_b_hp_tpl, chb_peaks_tpl, N_TEMPLATE, fs)
    cha_times_tpl = find_cha_initial_arrivals(ch_a_hp_tpl, chb_peaks_tpl, args.search_ms, fs, N_TEMPLATE)
    ch_a_template, _ = build_template(ch_a_hp_tpl, cha_times_tpl, N_TEMPLATE, fs)
    cha_offset_samp = int(np.median(
        [a - b for a, b in zip(cha_times_tpl[:N_TEMPLATE], chb_peaks_tpl[:N_TEMPLATE])]
    ))
    R_auto_a = correlate(ch_a_template, ch_a_template, mode='full')
    R_auto_b = correlate(ch_b_template, ch_b_template, mode='full')
    print(f"  templates rebuilt (|Ch A|={np.max(np.abs(ch_a_template))*1000:.1f} mV, "
          f"|Ch B|={np.max(np.abs(ch_b_template))*1000:.1f} mV)")

    # Group pings into 30-s chunks matching the pipeline
    t_start_s = t_pulse[0]
    t_end_s   = t_pulse[-1]
    n_chunks  = int(np.ceil((t_end_s - t_start_s) / CHUNK_S))
    stride_chunks = max(1, int(args.every_min * 60 / CHUNK_S))
    sel_chunks = list(range(0, n_chunks, stride_chunks))
    print(f"  {n_chunks} chunks  →  plotting {len(sel_chunks)} (every {stride_chunks * CHUNK_S:.0f}s)")

    # To get the ping-level CCF for the mid-chunk ping we must find that ping
    # and re-run matched filter on its WAV window.  Loop & plot.
    with sf.SoundFile(wav, 'r') as f_wav:
        n_total = f_wav.frames
        for ci in sel_chunks:
            ck_t0 = t_start_s + ci * CHUNK_S
            ck_t1 = ck_t0 + CHUNK_S
            mask  = (t_pulse >= ck_t0) & (t_pulse < ck_t1)
            idxs  = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            i0, i1 = int(idxs[0]), int(idxs[-1]) + 1
            # pick ping nearest to chunk midpoint
            mid_i = int(idxs[len(idxs)//2])
            t_mid = t_pulse[mid_i]
            mid_samp = int(round(t_mid * fs))

            # Read a window around that ping covering Ch A arrival
            half = win_samples // 2
            margin = int(0.05 * fs)      # 50 ms safety margin
            lo = max(0, mid_samp - margin)
            hi = min(n_total, mid_samp + int(args.search_ms[1] * 1e-3 * fs) + half + margin)
            f_wav.seek(lo)
            seg = f_wav.read(hi - lo, dtype='int16')
            ch_a_v = seg[:, 0].astype(np.float32) / 32768 * args.ch_a_fs_v
            ch_b_v = seg[:, 1].astype(np.float32) / 32768 * args.ch_b_fs_v
            ch_a_hp = hp_filter(ch_a_v, fs)
            ch_b_hp = hp_filter(ch_b_v, fs)
            # local offset of mid-ping within the loaded slice
            local_chb_samp = mid_samp - lo

            res = analyze_ping_pair(ch_a_hp, ch_b_hp, local_chb_samp,
                                     ch_a_template, ch_b_template,
                                     R_auto_a, R_auto_b, fs, win_samples,
                                     cha_offset_samp, return_ccf=True)
            if res is None:
                continue

            cd = dict(
                chunk_idx = ci,
                ping_i_start = i0,
                ping_i_end   = i1,
                t_start_s = float(ck_t0),
                t_end_s   = float(ck_t1),
                ccf_a = res['ccf_a'], env_a = res['env_a'],
                ccf_b = res['ccf_b'], env_b = res['env_b'],
            )
            plot_per_chunk(cd, ch_a_template, ch_b_template, t_pulse,
                           tof_para, tof_shape, dtau_para, dtau_shape,
                           sig_para, sig_shape, inl_para, inl_shape, fs,
                           out_chunk_dir, wav_stem, tof_sens)
    print(f"  wrote plots → {out_chunk_dir}")
