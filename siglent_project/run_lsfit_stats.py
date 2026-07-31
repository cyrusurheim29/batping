#!/usr/bin/env python3
"""
run_lsfit_stats.py

Run Stubbs LS shape-fit analysis on all statistics campaign recordings.
By default uses vtest pipeline (±700 μs search radius).
Pass --dechirp to use vtest-dechirp (dechirped LS fit, Category 1 fix).

Usage:
    python3 run_lsfit_stats.py
    python3 run_lsfit_stats.py --dechirp
    python3 run_lsfit_stats.py --recordings-dir recordings/
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirp_stubbs_analysis as csa

# a0.3_sm_4ms chirp parameters for dechirp
F_LO_HZ       = 38450.0        # start frequency (Hz)
CHIRP_RATE    = 1_500_000.0    # BW/T = 6000 Hz / 4e-3 s (Hz/s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recordings-dir', default='recordings')
    ap.add_argument('--rep-rate', type=float, default=52.631578947368425)
    ap.add_argument('--n-template', type=int, default=100)
    ap.add_argument('--dechirp', action='store_true',
                    help='Use dechirped LS fit (vtest-dechirp, Category 1 fix)')
    args = ap.parse_args()

    rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           args.recordings_dir)

    wavs = sorted(glob.glob(
        os.path.join(rec_dir, '*stats_*', 'a0.3_sm_4ms_received.wav')))

    if not wavs:
        print(f"No WAVs found under {rec_dir}/*stats_*/")
        sys.exit(1)

    pipeline_name = 'stubbs_pipeline_vtest_dechirp' if args.dechirp else 'stubbs_pipeline_vtest'
    print(f"Found {len(wavs)} recordings  →  pipeline: {pipeline_name}\n")

    for i, wav in enumerate(wavs, 1):
        label = os.path.basename(os.path.dirname(wav))
        print(f"[{i:2d}/{len(wavs)}] {label}")
        rec_subdir   = os.path.dirname(wav)
        pipeline_dir = os.path.join(rec_subdir, pipeline_name)
        try:
            kwargs = dict(
                rep_rate=args.rep_rate,
                n_template=args.n_template,
                save_plot=True,
                _pipeline_dir=pipeline_dir,
                _search_radius_us=700.0,
            )
            if args.dechirp:
                kwargs['_dechirp_f_lo']  = F_LO_HZ
                kwargs['_dechirp_rate']  = CHIRP_RATE
            csa.analyze(wav, **kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()
