#!/usr/bin/env python3
"""
Long-term recording: a0.3_sm_4ms for 1 hour to characterise Δτ drift.
Waveform: α=0.3, sm BW=6 kHz (39–45 kHz), 4 ms chirp, 2.60 Vpp
Rep rate: 52.63 Hz  (~189k periods in 1 hr, WAV ~2.6 GB)
"""
import sys, os, time
import numpy as np
from datetime import datetime
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_sweep as cs

RECORD_HOURS = 1.0          # change to run shorter / longer
ALPHA        = 0.3
F0, F1       = 39_000, 45_000
DUR_MS       = 4.0
DEAD_MS      = 15.0
AMPL_VPP     = 2.60         # calibrated to ~90% FS

SDG_SR=400000; SDG_MAX=16384
buf_n   = min(int((DUR_MS+DEAD_MS)/1000*SDG_SR), SDG_MAX)
rep_rate = SDG_SR / buf_n
record_s = RECORD_HOURS * 3600.0
n_periods_est = int(record_s * rep_rate)

print(f"Waveform: α={ALPHA}  f={F0/1000:.0f}–{F1/1000:.0f} kHz  T={DUR_MS:.0f} ms")
print(f"Rep rate: {rep_rate:.2f} Hz  |  Record: {RECORD_HOURS:.1f} hr = {record_s:.0f} s")
print(f"Expected periods: {n_periods_est:,}  |  Vpp: {AMPL_VPP} V")
print(f"WAV size: ~{record_s*cs.DODOMIC_SAMPLE_RATE*2/1e9:.2f} GB")
print()

ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(cs.OUTPUT_DIR, 'recordings', f'{ts}_longterm_a0.3_sm_4ms')
os.makedirs(run_dir, exist_ok=True)
wav_path = os.path.join(run_dir, 'a0.3_sm_4ms_1hr_received.wav')
print(f"Saving to: {wav_path}")

piezo = cs.load_piezo_response()
_, chirp, _ = cs.generate_lfm_chirp(
    F0, F1, DUR_MS, cs.DODOMIC_SAMPLE_RATE,
    amplitude_shape='alpha_comp', piezo_response=piezo,
    window='tukey', alpha=ALPHA)
sdg_buf, _ = cs.create_sdg_buffer(chirp, DUR_MS, dead_ms=DEAD_MS)

pa, mic_idx = cs.find_ultramic()
if pa is None: sys.exit(1)

sdg = cs.SDG1062X_LAN()
sdg.connect()
print(f"SDG: {sdg.query('*IDN?')}")
sdg.set_arb_mode()
sdg.upload_waveform('LT_A03SM4', sdg_buf, rep_rate, ampl_vpp=AMPL_VPP)
sdg.output_on()
time.sleep(0.5)
print(f"\nSDG running. Recording for {RECORD_HOURS:.1f} hr...")

received = cs.record_audio(pa, mic_idx, record_s)

sdg.output_off(); sdg.disconnect(); pa.terminate()

wavfile.write(wav_path, cs.DODOMIC_SAMPLE_RATE,
              (received * 32767).astype(np.int16))
print(f"\nSaved: {wav_path}  ({os.path.getsize(wav_path)/1e9:.2f} GB)")
print(f"\nTo analyse:")
print(f"  python3 -c \"")
print(f"  import chirp_stubbs_analysis as st")
print(f"  r = st.analyze('{wav_path}', rep_rate={rep_rate:.4f}, output_dir='{run_dir}')")
print(f"  print(r['std_delta_tau_us'], 'μs')\"")
