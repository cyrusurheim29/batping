#!/usr/bin/env python3
"""
picoscope_batping_record.py

Continuous BATPING recording via PicoScope 5444D + ECM999.

Signal chain:
  SDG Ch1 → piezo array  (a0.3_sm_4ms chirp, 52.63 Hz rep rate)
  ECM999 → phantom box → SR560 → PicoScope Ch A

Records Ch A continuously for DURATION_MIN minutes using repeated block
captures, stitches to a single WAV file, then exits.  The WAV can be
fed directly to stubbs_longterm_chunked.py for Stubbs LS analysis.

Usage:
    python3 picoscope_batping_record.py
"""

import numpy as np
import ctypes, time, os, socket
from scipy.io import wavfile

# Ensure PicoSDK shared library is found on macOS
_pico_lib = '/Library/Frameworks/PicoSDK.framework/Libraries/libps5000a'
if os.path.isdir(_pico_lib) and _pico_lib not in os.environ.get('DYLD_LIBRARY_PATH',''):
    os.environ['DYLD_LIBRARY_PATH'] = _pico_lib + ':' + os.environ.get('DYLD_LIBRARY_PATH','')

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

SDG_IP   = "10.11.13.230"
SDG_PORT = 5025

# Waveform: a0.3_sm_4ms  (best ECM999 performer — 1.8 ns, δ(Δτ))
# α=0.3, 6 kHz BW (37–43 kHz), 4 ms burst, 15 ms dead → 52.63 Hz
BURST_MS   = 4.0
DEAD_MS    = 15.0
ALPHA      = 0.3
F_LO       = 37_000.0      # Hz  chirp start
F_HI       = 43_000.0      # Hz  chirp end
SDG_SR     = 400_000       # Sa/s SDG sample rate
SDG_AMPL   = 3.2           # Vpp  (calibrated 2026-07-27: 77.7% of ±2V range)

# Recording (overridden by --duration flag)
DURATION_MIN   = 15        # default recording length (minutes)
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))

# PicoScope  (Ch A — ECM999 via SR560;  Ch B — TMP36 temperature sensor)
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
TIMEBASE      = 64             # → 1.025 MS/s, 25.6 samples/cycle at 40 kHz
BLOCK_PERIODS = 200            # SDG periods captured per PicoScope block
CH_A_RANGE    = ps.PS5000A_RANGE["PS5000A_2V"]
CH_A_COUPLING = ps.PS5000A_COUPLING["PS5000A_AC"]
# TMP36 on Ch B: DC, ±2V range (output 0.1–2.0V over -40–150°C)
# Conversion: T(°C) = (V_mV − 500) / 10
CH_B_RANGE    = ps.PS5000A_RANGE["PS5000A_2V"]
CH_B_COUPLING = ps.PS5000A_COUPLING["PS5000A_DC"]
TMP36_V0_MV   = 500.0   # mV at 0°C
TMP36_SCALE   = 10.0    # mV/°C

# ═══════════════════════════════════════════════════════════════════════════
# SDG control
# ═══════════════════════════════════════════════════════════════════════════

class SDG:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((SDG_IP, SDG_PORT))

    def write(self, cmd):
        self.sock.sendall((cmd + '\n').encode())
        time.sleep(0.05)

    def query(self, cmd):
        self.write(cmd)
        return self.sock.recv(4096).decode().strip()

    def write_raw(self, data):
        self.sock.sendall(data)

    def upload_chirp(self, waveform='a0.3_sm_4ms', ampl_vpp=None):
        """Generate and upload a waveform to SDG Ch1.

        waveform : 'a0.3_sm_4ms'  — standard heat-test chirp (default)
                   'flat3db'       — flat LFM across -3dB BW of ECM999+SR560 H(f)
        """
        from scipy.signal import windows as _win, savgol_filter as _sg
        import glob as _glob

        burst_ms = BURST_MS; dead_ms = DEAD_MS
        buf_n    = min(int((burst_ms + dead_ms) / 1000 * SDG_SR), 16384)
        rep_hz   = SDG_SR / buf_n
        burst_n  = int(burst_ms / 1000 * SDG_SR)
        t_b      = np.arange(burst_n) / SDG_SR
        vpp      = ampl_vpp if ampl_vpp is not None else SDG_AMPL

        if waveform == 'a0.3_sm_4ms':
            f_lo, f_hi = F_LO, F_HI          # 38450–44450 Hz, 6 kHz BW
            phase  = 2*np.pi*(f_lo*t_b + (f_hi-f_lo)/(2*burst_ms/1000)*t_b**2)
            f_inst = f_lo + (f_hi-f_lo)/(burst_ms/1000)*t_b
            _bswv = os.path.join(OUT_DIR, 'ecm999_stepped_sine_bswv_20260716_013405.npz')
            _d    = np.load(_bswv)
            _freqs = _d['freqs'].astype(float)
            from scipy.signal import savgol_filter as _sg2
            _mag  = 10**(_sg2(_d['H_mag'], window_length=71, polyorder=3) / 20.0)
            _mag_f = np.clip(np.interp(f_inst, _freqs, _mag), 0.05, None)
            amp   = _mag_f ** (-ALPHA)
            amp   = amp / np.max(amp)
            burst = amp * np.sin(phase) * _win.tukey(burst_n, alpha=0.05)
            tag   = 'a0.3_sm_4ms'

        elif waveform == 'a0.3_md_4ms':
            # 12 kHz BW centred on 41450 Hz, α=0.3 compensation via H(f)^{-0.3}
            f_lo, f_hi = 35_450.0, 47_450.0
            phase  = 2*np.pi*(f_lo*t_b + (f_hi-f_lo)/(2*burst_ms/1000)*t_b**2)
            f_inst = f_lo + (f_hi-f_lo)/(burst_ms/1000)*t_b
            # Load measured chain H(f) and apply alpha_comp: amp ∝ H(f)^{-0.3}
            _bswv = os.path.join(OUT_DIR, 'ecm999_stepped_sine_bswv_20260716_013405.npz')
            _d    = np.load(_bswv)
            _freqs = _d['freqs'].astype(float)
            from scipy.signal import savgol_filter as _sg2
            _mag  = 10**(_sg2(_d['H_mag'], window_length=71, polyorder=3) / 20.0)
            _mag_at_f = np.interp(f_inst, _freqs, _mag)
            _mag_at_f = np.clip(_mag_at_f, 0.05, None)   # -26 dB floor
            amp   = _mag_at_f ** (-0.3)
            amp   = amp / np.max(amp)
            burst = amp * np.sin(phase) * _win.tukey(burst_n, alpha=0.05)
            tag   = 'a0.3_md_4ms'
            if ampl_vpp is None:
                vpp = 3.2

        elif waveform in ('bw_20kHz_wide_4ms_diff', 'bw_10dB_asym_2ms_diff'):
            # Diffraction-compensated: 1/H(f) × (f/f_center) pre-emphasis
            if waveform == 'bw_20kHz_wide_4ms_diff':
                f_lo, f_hi, burst_ms = 31_475.0, 51_475.0, 4.0
            else:
                f_lo, f_hi, burst_ms = 37_200.0, 45_000.0, 2.0
            burst_n = int(burst_ms / 1000 * SDG_SR)
            t_b     = np.arange(burst_n) / SDG_SR
            phase   = 2*np.pi*(f_lo*t_b + (f_hi-f_lo)/(2*burst_ms/1000)*t_b**2)
            f_inst  = f_lo + (f_hi-f_lo)/(burst_ms/1000)*t_b
            f_center = (f_lo + f_hi) / 2.0
            _bswv = os.path.join(OUT_DIR, 'ecm999_stepped_sine_bswv_20260716_013405.npz')
            _d    = np.load(_bswv)
            _freqs = _d['freqs'].astype(float)
            from scipy.signal import savgol_filter as _sg2
            _mag  = 10**(_sg2(_d['H_mag'], window_length=71, polyorder=3) / 20.0)
            _mag_at_f = np.interp(f_inst, _freqs, _mag)
            diffraction = f_inst / f_center
            eff_mag = np.clip(_mag_at_f / diffraction, 0.20, None)
            amp     = 1.0 / eff_mag
            amp     = amp / np.max(amp)
            burst   = amp * np.sin(phase) * _win.tukey(burst_n, alpha=0.05)
            tag     = waveform
            if ampl_vpp is None:
                vpp = 3.2

        elif waveform == 'flat3db':
            # Load -3dB bandwidth from BSWV H(f) measurement
            _bswv = os.path.join(OUT_DIR, 'ecm999_stepped_sine_bswv_20260716_013405.npz')
            _d    = np.load(_bswv)
            _freqs  = _d['freqs'].astype(float)
            _mag_db = _sg(_d['H_mag'], window_length=71, polyorder=3)
            _above  = _mag_db >= _mag_db.max() - 3.0
            _idxs   = np.where(_above)[0]
            f_lo = float(_freqs[_idxs[0]]); f_hi = float(_freqs[_idxs[-1]])
            phase  = 2*np.pi*(f_lo*t_b + (f_hi-f_lo)/(2*burst_ms/1000)*t_b**2)
            burst  = np.sin(phase) * _win.tukey(burst_n, alpha=0.1)
            tag    = f'flat3db_{f_lo/1e3:.1f}_{f_hi/1e3:.1f}kHz'

        else:
            raise ValueError(f"Unknown waveform '{waveform}'. Use 'a0.3_sm_4ms' or 'flat3db'.")

        buf = np.zeros(buf_n); buf[:burst_n] = burst
        mx  = np.max(np.abs(buf));
        if mx > 0: buf /= mx
        buf_i16  = (np.clip(buf, -1, 1) * 32767).astype('<i2')
        prefix   = f"C1:WVDT WVNM,BATPING,FREQ,{rep_hz:.6f},AMPL,{vpp},OFST,0,PHASE,0,WAVEDATA,".encode()
        self.write_raw(prefix + buf_i16.tobytes() + b'\n')
        time.sleep(0.8)
        self.write("C1:ARWV NAME,BATPING")
        time.sleep(0.3)
        self.write("C1:BSWV WVTP,ARB")
        time.sleep(0.2)
        print(f"  SDG: uploaded {tag}  rep={rep_hz:.3f} Hz  {vpp} Vpp  buf={buf_n} samples")
        return rep_hz, tag

    def output_on(self):  self.write("C1:OUTP ON")
    def output_off(self): self.write("C1:OUTP OFF")
    def close(self):      self.sock.close()


# ═══════════════════════════════════════════════════════════════════════════
# PicoScope helpers
# ═══════════════════════════════════════════════════════════════════════════

def open_scope():
    status  = {}
    chandle = ctypes.c_int16()
    status["open"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
    assert_pico_ok(status["open"])

    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    status["chA"] = ps.ps5000aSetChannel(chandle, chA, 1, CH_A_COUPLING, CH_A_RANGE, 0.0)
    assert_pico_ok(status["chA"])

    # Ch B — TMP36 temperature sensor (DC, ±2V)
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    status["chB"] = ps.ps5000aSetChannel(chandle, chB, 1, CH_B_COUPLING, CH_B_RANGE, 0.0)
    assert_pico_ok(status["chB"])

    # Disable unused channels C and D
    for ch_name in ["PS5000A_CHANNEL_C", "PS5000A_CHANNEL_D"]:
        ch = ps.PS5000A_CHANNEL[ch_name]
        ps.ps5000aSetChannel(chandle, ch, 0,
                             ps.PS5000A_COUPLING["PS5000A_DC"],
                             ps.PS5000A_RANGE["PS5000A_1V"], 0.0)

    max_s   = ctypes.c_int32(1000)
    dt_ns   = ctypes.c_float()
    status["tb"] = ps.ps5000aGetTimebase2(chandle, TIMEBASE, 1000,
                                           ctypes.byref(dt_ns),
                                           ctypes.byref(max_s), 0)
    assert_pico_ok(status["tb"])
    actual_dt  = float(dt_ns.value)
    actual_fs  = 1e9 / actual_dt
    print(f"  PicoScope: {actual_dt:.1f} ns/sample = {actual_fs/1e3:.1f} kSa/s")
    return chandle, actual_fs


def capture_block(chandle, n_samples):
    """Capture one block from Ch A and Ch B.
    Returns (ch_a_mv float32, tmp36_temp_C float) where tmp36_temp_C
    is the mean temperature over the block from the TMP36 on Ch B.
    """
    chA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    chB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    no_trig = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_NONE"]
    ps.ps5000aSetSimpleTrigger(chandle, 0, chA, 0, no_trig, 0, 5)
    ps.ps5000aRunBlock(chandle, 0, n_samples, TIMEBASE, None, 0, None, None)
    ready = ctypes.c_int16(0)
    while ready.value == 0:
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    buf_a = (ctypes.c_int16 * n_samples)()
    buf_b = (ctypes.c_int16 * n_samples)()
    ratio = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
    ps.ps5000aSetDataBuffer(chandle, chA, ctypes.byref(buf_a), n_samples, 0, ratio)
    ps.ps5000aSetDataBuffer(chandle, chB, ctypes.byref(buf_b), n_samples, 0, ratio)

    overflow = ctypes.c_int16()
    n_ret    = ctypes.c_uint32(n_samples)
    ps.ps5000aGetValues(chandle, 0, ctypes.byref(n_ret), 1, 0, 0,
                        ctypes.byref(overflow))

    maxADC = ctypes.c_int16()
    ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))

    mv_a = np.array(adc2mV(buf_a, CH_A_RANGE, maxADC), dtype=np.float32)
    mv_b = np.array(adc2mV(buf_b, CH_B_RANGE, maxADC), dtype=np.float32)

    # TMP36: T(°C) = (V_mV − 500) / 10,  averaged over the whole block
    tmp36_temp_C = float((np.mean(mv_b) - TMP36_V0_MV) / TMP36_SCALE)

    if overflow.value & 0x01:
        print("  WARNING: Ch A overflow — reduce SR560 gain or use PS5000A_5V")
    if overflow.value & 0x02:
        print("  WARNING: Ch B overflow — check TMP36 wiring")
    return mv_a, tmp36_temp_C


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=DURATION_MIN,
                        help='Recording duration in minutes (default 15)')
    parser.add_argument('--label', type=str, default='heattest',
                        help='Label appended to recording directory name')
    parser.add_argument('--waveform', type=str, default='a0.3_sm_4ms',
                        choices=['a0.3_sm_4ms', 'flat3db'],
                        help='Waveform to upload: a0.3_sm_4ms (default) or flat3db (-3dB BW, no compensation)')
    parser.add_argument('--ampl', type=float, default=None,
                        help='SDG amplitude in Vpp (default: stored calibration value)')
    parser.add_argument('--adc-range', type=str, default='2V',
                        choices=['200mV','500mV','1V','2V','5V','10V','20V','50V'],
                        help='PicoScope Ch A voltage range (default: 2V = ±2V)')
    args = parser.parse_args()
    duration_min = args.duration

    print("=" * 60)
    print(f"BATPING recording  —  {args.waveform}  (ECM999 + PicoScope)")
    print(f"Duration: {duration_min} min    Ch A: ±2V AC")
    print("=" * 60)

    # ── ADC range ─────────────────────────────────────────────────────────
    global CH_A_RANGE, SCALE
    _range_key = f"PS5000A_{args.adc_range.upper()}"
    _range_mv  = {'200MV':200,'500MV':500,'1V':1000,'2V':2000,
                  '5V':5000,'10V':10000,'20V':20000,'50V':50000}
    _fs_mv     = _range_mv.get(args.adc_range.upper(), 2000)
    CH_A_RANGE = ps.PS5000A_RANGE[_range_key]
    print(f"  ADC range: ±{args.adc_range}  (full scale ±{_fs_mv} mV)")

    # ── SDG setup ─────────────────────────────────────────────────────────
    print("\nConnecting to SDG...")
    sdg = SDG()
    print(f"  {sdg.query('*IDN?')[:55]}")
    rep_hz, wf_tag = sdg.upload_chirp(waveform=args.waveform, ampl_vpp=args.ampl)
    sdg.output_on()
    time.sleep(0.5)   # let piezo settle

    # ── PicoScope setup ───────────────────────────────────────────────────
    print("\nOpening PicoScope...")
    chandle, actual_fs = open_scope()

    # Block size: cover BLOCK_PERIODS SDG repetitions
    period_samples = int(actual_fs / rep_hz)
    n_samples      = BLOCK_PERIODS * period_samples
    block_duration = n_samples / actual_fs
    total_blocks   = int(np.ceil(duration_min * 60 / block_duration))

    print(f"  Block: {BLOCK_PERIODS} periods = {n_samples:,} samples = {block_duration:.2f}s")
    print(f"  Total blocks: {total_blocks}  ({total_blocks * block_duration / 60:.1f} min)")

    import datetime, soundfile as _sf
    stamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_wav = os.path.join(OUT_DIR, 'recordings',
                           f'{stamp}_batping_{args.label}',
                           f'{wf_tag}_received.wav')
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    fs_int  = int(round(actual_fs))

    # Fixed normalisation: ±2 V full scale → int16.
    # soundfile writes RF64 WAV automatically for files > 4 GB,
    # so 35 min @ 1025 kHz (4.3 GB) is handled without any size limit.
    SCALE    = 32767.0 / _fs_mv   # normalise to int16 using selected ADC full scale
    fs_write = int(round(actual_fs))

    out_tmp36 = out_wav.replace('_received.wav', '_tmp36.csv')
    print(f"\nRecording → {out_wav}")
    print(f"TMP36 CSV → {out_tmp36}\n")
    t_start    = time.time()
    n_written  = 0
    tmp36_rows = []   # (elapsed_s, temp_C) per block

    # RF64 WAV supports files >4 GB (standard WAV caps at 2^31 samples = ~35 min)
    sf_out = _sf.SoundFile(out_wav, mode='w', samplerate=fs_write,
                           channels=1, subtype='PCM_16', format='RF64')
    KEEPALIVE_BLOCKS = 8   # ping SDG every ~8 blocks (~30 s) to prevent timeout
    try:
        for i in range(total_blocks):
            mv_a, temp_C = capture_block(chandle, n_samples)
            sf_out.write((mv_a * SCALE).astype(np.int16))
            n_written += len(mv_a)

            elapsed = time.time() - t_start
            done    = (i + 1) / total_blocks * 100
            peak_mv = float(np.max(np.abs(mv_a)))
            audio_s = (i + 1) * block_duration   # exact audio time at end of this block
            tmp36_rows.append((elapsed, audio_s, temp_C))
            print(f"  Block {i+1:4d}/{total_blocks}  "
                  f"{elapsed/60:.2f} min  {done:5.1f}%  "
                  f"peak={peak_mv:.1f} mV  TMP36={temp_C:.3f}°C", flush=True)

            if (i + 1) % KEEPALIVE_BLOCKS == 0:
                try:
                    sdg.query('*IDN?')   # keep TCP connection alive
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        sf_out.close()
        # Save TMP36 CSV before closing scope
        import csv as _csv
        with open(out_tmp36, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['elapsed_s', 'audio_s', 'tmp36_temp_C'])
            w.writerows(tmp36_rows)
        print(f"  TMP36 CSV saved: {out_tmp36}  ({len(tmp36_rows)} rows)")
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)
        sdg.output_off()
        sdg.close()

    elapsed = time.time() - t_start
    size_gb = os.path.getsize(out_wav) / 1e9
    print(f"\nSaved: {out_wav}")
    print(f"  Duration: {elapsed/60:.2f} min  |  {n_written:,} samples  |  {fs_write} Hz  |  {size_gb:.2f} GB")
    print(f"\nTo analyse:")
    print(f"  python3 stubbs_longterm_chunked.py {out_wav} --chunk-sec 30")


if __name__ == '__main__':
    main()
