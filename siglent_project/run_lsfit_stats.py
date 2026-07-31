#!/usr/bin/env python3
"""
run_lsfit_stats.py

Run Stubbs LS shape-fit analysis on all statistics campaign recordings,
generating stubbs_lsfit PNGs in stubbs_pipeline_v3/ next to each WAV.

Usage:
    python3 run_lsfit_stats.py
    python3 run_lsfit_stats.py --recordings-dir recordings/
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as csa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recordings-dir', default='recordings')
    ap.add_argument('--rep-rate', type=float, default=52.631578947368425)
    ap.add_argument('--n-template', type=int, default=100)
    args = ap.parse_args()

    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           args.recordings_dir)

    wavs = sorted(glob.glob(
        os.path.join(rec_dir, '*stats_*', 'a0.3_sm_4ms_received.wav')))

    if not wavs:
        print(f"No WAVs found under {rec_dir}/*stats_*/")
        sys.exit(1)

    print(f"Found {len(wavs)} recordings\n")

    for i, wav in enumerate(wavs, 1):
        label = os.path.basename(os.path.dirname(wav))
        print(f"[{i:2d}/{len(wavs)}] {label}")
        # Put the PNG inside the recording's own subdir, not the shared parent
        rec_subdir = os.path.dirname(wav)
        pipeline_dir = os.path.join(rec_subdir, 'stubbs_pipeline_vtest')
        try:
            csa.analyze(wav, rep_rate=args.rep_rate,
                        n_template=args.n_template, save_plot=True,
                        _pipeline_dir=pipeline_dir,
                        _search_radius_us=700.0)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()
