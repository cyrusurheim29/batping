#!/usr/bin/env python3
"""Send a few periods of 40 kHz sine bursts and record 3s to test mic position."""
import socket, time
import numpy as np
import pyaudio
from scipy.io import wavfile

SR_SDG = 400000
SR_MIC = 384000
F0     = 40000
DUR_MS = 1.0
DEAD_MS = 15.0

# Build SDG buffer: 1ms sine burst + 15ms dead = 62.5 Hz rep
t = np.linspace(0, DUR_MS/1000, int(DUR_MS/1000*SR_SDG), endpoint=False)
burst = 0.9 * np.sin(2*np.pi*F0*t)
buf = np.zeros(int((DUR_MS + DEAD_MS)/1000*SR_SDG))
buf[:len(burst)] = burst
rep = SR_SDG / len(buf)
print(f"Rep rate: {rep:.1f} Hz  |  Period: {len(buf)} samples")

# Upload to SDG
d = (np.clip(buf, -1, 1) * 32767).astype(np.int16).astype('<i2').tobytes()
s = socket.socket()
s.settimeout(10)
s.connect(('10.11.13.230', 5025))
cmd = f'C1:WVDT WVNM,SINE40K,FREQ,{rep:.4f},AMPL,2.0,OFST,0,PHASE,0,WAVEDATA,'.encode()
s.sendall(cmd + d + b'\n')
time.sleep(0.6)
s.sendall(b'C1:ARWV NAME,SINE40K\n')
time.sleep(0.3)
s.sendall(b'C1:OUTP ON\n')
time.sleep(0.3)
print("SDG running.")

# Record 3s
pa = pyaudio.PyAudio()
mic = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if 'UltraMic' in info['name'] and info['maxInputChannels'] > 0:
        mic = i
        break
if mic is None:
    print("UltraMic not found!"); s.sendall(b'C1:OUTP OFF\n'); s.close(); pa.terminate(); exit(1)
print(f"UltraMic at index {mic}")

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SR_MIC,
                 input=True, input_device_index=mic, frames_per_buffer=4096)
frames = []
n = int(3.0 * SR_MIC)
while n > 0:
    ch = min(4096, n)
    frames.append(stream.read(ch, exception_on_overflow=False))
    n -= ch
stream.stop_stream()
stream.close()

s.sendall(b'C1:OUTP OFF\n')
s.close()
pa.terminate()

sig = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / 32768.0
wavfile.write('sine40k_mic_test.wav', SR_MIC, (sig * 32767).astype(np.int16))
print("Saved: sine40k_mic_test.wav")

# Quick analysis: show incident vs reflected peak amplitudes in first period
from scipy.signal import butter, sosfiltfilt, hilbert
sos = butter(8, 30000/(SR_MIC/2), btype='high', output='sos')
sig_hp = sosfiltfilt(sos, sig)
pn = int(SR_MIC / rep)
seg = sig_hp[pn:2*pn]  # second period (settled)
env = np.abs(hilbert(seg))
t_ms = np.arange(len(seg)) / SR_MIC * 1000
inc_peak  = np.max(env[:int(0.004*SR_MIC)])    # first 4ms = incident
refl_peak = np.max(env[int(0.004*SR_MIC):int(0.010*SR_MIC)])  # 4-10ms = reflected
print(f"\nIncident peak:  {inc_peak:.4f}")
print(f"Reflected peak: {refl_peak:.4f}")
print(f"Ratio refl/inc: {refl_peak/inc_peak:.3f}  (want close to 1.0)")
