#!/usr/bin/env python3
"""Record bw_6kHz_flat in PPP Mode B (stream restart per ping) — eliminates drift accumulation."""
import sys, os, time
import numpy as np
from datetime import datetime
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import chirp_sweep as cs

N_PERIODS  = 500       # 500 independent pings
SETTLE_S   = 2.0       # settle time after stream open
AMPL_VPP   = 2.15      # calibrated voltage for flat 6kHz
F0, F1     = 38475, 44475
DUR_MS     = 2.0
AMP_SHAPE  = 'flat'
LABEL      = 'bw_6kHz_flat_pppB'

SDG_SR = 400000; SDG_MAX = 16384
buf_samples = min(int((DUR_MS+15)/1000*SDG_SR), SDG_MAX)
rep_rate    = SDG_SR / buf_samples
print(f"Rep rate: {rep_rate:.2f} Hz  |  Period: {buf_samples/SDG_SR*1000:.1f} ms")
print(f"N periods: {N_PERIODS}  |  Settle: {SETTLE_S}s  |  Vpp: {AMPL_VPP}")
print(f"Estimated time: {N_PERIODS*(SETTLE_S+buf_samples/SDG_SR)/60:.1f} min")

ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(cs.OUTPUT_DIR, 'recordings', f'{ts}_flat6kHz_pppB')
raw_dir = os.path.join(run_dir, 'raw')
os.makedirs(raw_dir, exist_ok=True)

piezo_response = cs.load_piezo_response()
_, chirp, _ = cs.generate_lfm_chirp(F0, F1, DUR_MS, cs.DODOMIC_SAMPLE_RATE,
                                     AMP_SHAPE, piezo_response, window='tukey')
sdg_buf, _ = cs.create_sdg_buffer(chirp, DUR_MS)

pa, mic_index = cs.find_ultramic()
if pa is None: sys.exit(1)

sdg = cs.SDG1062X_LAN()
sdg.connect()
print(f"SDG: {sdg.query('*IDN?')}")
sdg.set_arb_mode()

wf_name = 'FLAT6K'
sdg.upload_waveform(wf_name, sdg_buf, rep_rate, ampl_vpp=AMPL_VPP)
sdg.output_on()
time.sleep(0.3)
print(f"\nSDG running. Starting PPP Mode B recording...")

periods = cs.record_period_by_period(pa, mic_index, rep_rate, N_PERIODS,
                                     mode='B', settle_s=SETTLE_S)
sdg.output_off()
sdg.disconnect()
pa.terminate()

# Save as single concatenated WAV + individual periods
concat = periods.flatten()
wav_path = os.path.join(raw_dir, f'{LABEL}_received.wav')
wavfile.write(wav_path, cs.DODOMIC_SAMPLE_RATE, (concat * 32767).astype(np.int16))
print(f"\nSaved: {wav_path}")
print(f"Shape: {periods.shape}  ({periods.shape[0]} periods × {periods.shape[1]} samples)")
print(f"\nNow run Stubbs analysis:")
print(f"  python run_stubbs_july8.py  (add this wav to the list)")
