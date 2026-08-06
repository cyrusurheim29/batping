"""
WaveStation 2052 Interface Module

Provides communication with Teledyne LeCroy WaveStation 2052 arbitrary waveform generator
via USB (USBTMC protocol) using pyvisa.

Key capabilities:
- Connect/disconnect to device
- Upload arbitrary waveforms to device memory
- Configure output parameters (amplitude, frequency, offset)
- Generate standard pulse shapes (Gaussian, chirp, etc.)
"""

import numpy as np
import pyvisa
from typing import Optional, Tuple
import time


class WaveStation:
    """Interface to Teledyne LeCroy WaveStation 2052."""

    # WaveStation 2052 specs (WaveStation 2000 series uses 16k limit)
    MAX_ARB_POINTS = 16384  # Maximum points per arbitrary waveform
    MIN_ARB_POINTS = 32     # Minimum points
    DAC_BITS = 14           # 14-bit DAC resolution
    MAX_SAMPLE_RATE = 125e6  # 125 MSa/s max

    # User-defined waveform memory slots for WaveStation 2000 series
    # M50-M59 are user slots (M0-M49 are built-in waveforms)
    USER_MEMORY_SLOTS = list(range(50, 60))  # M50 through M59
    DEFAULT_USER_SLOT = 50  # Default to M50

    def __init__(self, resource_addr: Optional[str] = None):
        """
        Initialize WaveStation interface.

        Args:
            resource_addr: VISA resource address. If None, auto-detect.
        """
        self.rm = pyvisa.ResourceManager('@py')
        self.resource_addr = resource_addr
        self.device = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to the WaveStation."""
        if self.connected:
            return True

        # Auto-detect if no address provided
        if self.resource_addr is None:
            resources = self.rm.list_resources()
            lecroy_resources = [r for r in resources if '1535' in r or 'LCRY' in r]
            if not lecroy_resources:
                raise ConnectionError("No LeCroy WaveStation found on USB")
            self.resource_addr = lecroy_resources[0]

        try:
            self.device = self.rm.open_resource(self.resource_addr)
            self.device.timeout = 10000  # 10 second timeout

            # Verify identity
            idn = self.device.query("*IDN?")
            if "WaveStation" not in idn:
                raise ConnectionError(f"Device is not a WaveStation: {idn}")

            self.connected = True
            print(f"Connected to: {idn.strip()}")
            return True

        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Failed to connect: {e}")

    def disconnect(self):
        """Disconnect from the WaveStation."""
        if self.device:
            self.device.close()
        self.connected = False
        self.device = None

    def _check_connection(self):
        """Ensure device is connected."""
        if not self.connected or self.device is None:
            raise RuntimeError("Not connected to WaveStation. Call connect() first.")

    def query(self, cmd: str) -> str:
        """Send query and return response."""
        self._check_connection()
        return self.device.query(cmd).strip()

    def write(self, cmd: str):
        """Send command without expecting response."""
        self._check_connection()
        self.device.write(cmd)

    def get_channel_status(self, channel: int = 1) -> dict:
        """Get current channel configuration."""
        self._check_connection()

        output = self.query(f"C{channel}:OUTP?")
        bswv = self.query(f"C{channel}:BSWV?")

        return {
            'output': output,
            'basic_wave': bswv
        }

    def set_output(self, channel: int = 1, enable: bool = True, load: str = "HZ"):
        """
        Enable/disable channel output.

        Args:
            channel: Channel number (1 or 2)
            enable: True to enable output
            load: Load impedance ("50" for 50 ohm, "HZ" for high-Z)
        """
        state = "ON" if enable else "OFF"
        self.write(f"C{channel}:OUTP {state},LOAD,{load}")

    def set_basic_wave(self, channel: int = 1, waveform: str = "SINE",
                       freq_hz: float = 1000, amp_vpp: float = 1.0,
                       offset_v: float = 0.0):
        """
        Configure basic waveform output.

        Args:
            channel: Channel number
            waveform: SINE, SQUARE, RAMP, PULSE, NOISE, ARB, DC
            freq_hz: Frequency in Hz
            amp_vpp: Peak-to-peak amplitude in volts
            offset_v: DC offset in volts
        """
        cmd = f"C{channel}:BSWV WVTP,{waveform},FRQ,{freq_hz}HZ,AMP,{amp_vpp}V,OFST,{offset_v}V"
        self.write(cmd)

    def upload_arbitrary_waveform(self, waveform: np.ndarray, name: str = "wave1",
                                  memory_slot: Optional[int] = None,
                                  amp_vpp: float = 1.0) -> int:
        """
        Upload arbitrary waveform to device memory using WVCSV command.

        Args:
            waveform: Numpy array of waveform samples (normalized -1 to 1)
            name: Name for the waveform in device memory
            memory_slot: Memory slot M50-M59 (default: M50). Must be user slot.
            amp_vpp: Peak-to-peak amplitude in volts for scaling waveform data

        Returns:
            Memory slot index where waveform was stored
        """
        self._check_connection()

        if memory_slot is None:
            memory_slot = self.DEFAULT_USER_SLOT

        if memory_slot not in self.USER_MEMORY_SLOTS:
            raise ValueError(f"Memory slot must be M50-M59, got M{memory_slot}")

        # Ensure correct length
        n_points = len(waveform)
        if n_points < self.MIN_ARB_POINTS:
            pad_length = self.MIN_ARB_POINTS - n_points
            waveform = np.pad(waveform, (0, pad_length), mode='constant')
            n_points = self.MIN_ARB_POINTS
        elif n_points > self.MAX_ARB_POINTS:
            indices = np.linspace(0, n_points - 1, self.MAX_ARB_POINTS).astype(int)
            waveform = waveform[indices]
            n_points = self.MAX_ARB_POINTS

        # Normalize to [-1, 1] then scale to voltage
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # Scale to voltage values (WVCSV expects amplitude in Volts)
        # Scale to half of amp_vpp since waveform is -1 to 1
        waveform_volts = waveform * (amp_vpp / 2.0)

        # Build CSV data string with header "Amplitude" followed by values
        # Format: Amplitude,val1,val2,val3,...
        data_values = ','.join(f'{v:.6f}' for v in waveform_volts)
        csv_data = f'Amplitude,{data_values}'

        # WVCSV command syntax (from SCPI manual):
        # WVCSV <address>,WAVENM,<wavename>,CSVLENG,<csvlength>,CSVDATA,<csvdata>
        cmd = f'WVCSV M{memory_slot},WAVENM,{name},CSVLENG,{n_points},CSVDATA,{csv_data}'

        self.write(cmd)
        time.sleep(0.5)  # Give device time to process (larger waveforms need more time)

        print(f"Uploaded {n_points} points as '{name}' to M{memory_slot}")
        return memory_slot

    def list_stored_waveforms(self) -> str:
        """
        Query all waveform names stored in device memory.

        Returns:
            String containing all memory slots and their waveform names.
            User slots (M50-M59) show "EMPTY" if no waveform stored.
        """
        return self.query("STL?")

    def set_arbitrary_waveform(self, channel: int = 1, memory_slot: int = 50,
                               freq_hz: float = 1000, amp_vpp: float = 1.0,
                               offset_v: float = 0.0):
        """
        Configure channel to output an arbitrary waveform from memory.

        Args:
            channel: Channel number (1 or 2)
            memory_slot: Memory slot index (M50-M59 for user waveforms)
            freq_hz: Output frequency in Hz (how fast to play the waveform)
            amp_vpp: Peak-to-peak amplitude in volts
            offset_v: DC offset in volts
        """
        # Set to ARB mode with parameters
        self.write(f"C{channel}:BSWV WVTP,ARB,FRQ,{freq_hz}HZ,AMP,{amp_vpp}V,OFST,{offset_v}V")
        # Select waveform by INDEX (not NAME - NAME only works for built-in waveforms)
        self.write(f"C{channel}:ARWV INDEX,{memory_slot}")

    def get_arb_status(self, channel: int = 1) -> str:
        """Query current arbitrary waveform selection for a channel."""
        return self.query(f"C{channel}:ARWV?")

    def trigger_burst(self, channel: int = 1, n_cycles: int = 1,
                      trigger_source: str = "INT", period_s: float = 0.1):
        """
        Configure burst mode for pulsed output.

        Args:
            channel: Channel number
            n_cycles: Number of waveform cycles per burst
            trigger_source: "INT" (internal), "EXT" (external), "MAN" (manual)
            period_s: Period between bursts (for internal trigger)
        """
        # Enable burst mode
        self.write(f"C{channel}:BTWV STATE,ON,TRSR,{trigger_source},GATE_NCYC,NCYC,TIME,{n_cycles},MTRIG,ON")
        if trigger_source == "INT":
            self.write(f"C{channel}:BTWV PRD,{period_s}S")


def generate_gaussian_pulse(n_points: int, sigma_frac: float = 0.15,
                           carrier_freq_cycles: float = 0) -> np.ndarray:
    """
    Generate a Gaussian-envelope pulse.

    Args:
        n_points: Number of sample points
        sigma_frac: Standard deviation as fraction of total length
        carrier_freq_cycles: Number of carrier cycles (0 for baseband pulse)

    Returns:
        Normalized waveform array
    """
    t = np.linspace(-0.5, 0.5, n_points)
    sigma = sigma_frac
    envelope = np.exp(-t**2 / (2 * sigma**2))

    if carrier_freq_cycles > 0:
        carrier = np.sin(2 * np.pi * carrier_freq_cycles * t)
        waveform = envelope * carrier
    else:
        waveform = envelope

    return waveform


def generate_chirp(n_points: int, f_start_cycles: float, f_end_cycles: float,
                   envelope: str = "rectangular") -> np.ndarray:
    """
    Generate a linear frequency chirp.

    Args:
        n_points: Number of sample points
        f_start_cycles: Starting frequency (in cycles over the waveform)
        f_end_cycles: Ending frequency (in cycles over the waveform)
        envelope: "rectangular", "gaussian", or "hanning"

    Returns:
        Normalized waveform array
    """
    t = np.linspace(0, 1, n_points)

    # Instantaneous frequency: f(t) = f0 + (f1-f0)*t
    # Phase: integral of 2*pi*f(t) = 2*pi*(f0*t + (f1-f0)*t^2/2)
    phase = 2 * np.pi * (f_start_cycles * t + (f_end_cycles - f_start_cycles) * t**2 / 2)
    waveform = np.sin(phase)

    # Apply envelope
    if envelope == "gaussian":
        env = np.exp(-((t - 0.5) ** 2) / (2 * 0.15**2))
        waveform *= env
    elif envelope == "hanning":
        env = 0.5 * (1 - np.cos(2 * np.pi * t))
        waveform *= env

    return waveform


def generate_shaped_pulse(n_points: int, carrier_cycles: float,
                         rise_frac: float = 0.1, fall_frac: float = 0.1) -> np.ndarray:
    """
    Generate a pulse with smooth rise/fall (raised cosine edges).

    Args:
        n_points: Number of sample points
        carrier_cycles: Number of carrier oscillation cycles
        rise_frac: Rise time as fraction of total length
        fall_frac: Fall time as fraction of total length

    Returns:
        Normalized waveform array
    """
    t = np.linspace(0, 1, n_points)

    # Create envelope with raised cosine edges
    envelope = np.ones(n_points)

    rise_samples = int(rise_frac * n_points)
    fall_samples = int(fall_frac * n_points)

    # Rise edge (half cosine)
    if rise_samples > 0:
        envelope[:rise_samples] = 0.5 * (1 - np.cos(np.pi * np.arange(rise_samples) / rise_samples))

    # Fall edge (half cosine)
    if fall_samples > 0:
        envelope[-fall_samples:] = 0.5 * (1 + np.cos(np.pi * np.arange(fall_samples) / fall_samples))

    # Carrier
    carrier = np.sin(2 * np.pi * carrier_cycles * t)

    return envelope * carrier


if __name__ == "__main__":
    # Quick test
    ws = WaveStation()
    ws.connect()

    print("\nChannel 1 status:", ws.get_channel_status(1))

    # List stored waveforms
    print("\nStored waveforms:")
    print(ws.list_stored_waveforms())

    # Generate a test pulse (40 cycles of carrier in Gaussian envelope)
    pulse = generate_gaussian_pulse(1024, sigma_frac=0.1, carrier_freq_cycles=40)

    print("\nUploading test Gaussian pulse to M50...")
    slot = ws.upload_arbitrary_waveform(pulse, name="TESTPULSE", memory_slot=50, amp_vpp=2.0)

    # Verify upload
    print("\nStored waveforms after upload:")
    print(ws.list_stored_waveforms())

    # Check ARB status
    print("\nARB status:", ws.get_arb_status(1))

    ws.disconnect()
    print("\nDone!")
