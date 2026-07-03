"""
Siglent SDG1000X Function Generator Interface

Provides USB/LAN SCPI control for the SDG1000X series including arbitrary waveform upload.
Ported from agilent_33220a.py with Siglent-specific command syntax.
"""

import numpy as np
import pyvisa
from typing import Optional, Literal


class SiglentSDG1000X:
    """Interface to Siglent SDG1000X series Function Generator."""

    VENDOR_ID = 0xF4EC  # Siglent USB vendor ID

    def __init__(self, channel: int = 1):
        self.rm: Optional[pyvisa.ResourceManager] = None
        self.instr: Optional[pyvisa.resources.Resource] = None
        self.resource_name: Optional[str] = None
        self.channel = channel  # Siglent uses channel prefixes (C1:, C2:)

    def _cmd(self, cmd: str) -> str:
        """Prepend channel prefix to command."""
        return f"C{self.channel}:{cmd}"

    def connect(self, resource_name: Optional[str] = None) -> bool:
        """
        Connect to the SDG1000X. Auto-detects if resource_name not provided.

        Args:
            resource_name: VISA resource string, e.g.:
                - USB: 'USB0::0xF4EC::0x1101::SDG1234::INSTR'
                - LAN: 'TCPIP0::192.168.1.100::inst0::INSTR'
        """
        self.rm = pyvisa.ResourceManager()

        if resource_name:
            self.resource_name = resource_name
        else:
            # Auto-detect by vendor ID
            resources = self.rm.list_resources()
            for res in resources:
                if f'{self.VENDOR_ID:04X}' in res.upper() or 'F4EC' in res.upper():
                    self.resource_name = res
                    break

            if not self.resource_name:
                raise RuntimeError(f"Siglent SDG not found. Available: {resources}")

        self.instr = self.rm.open_resource(self.resource_name)
        self.instr.timeout = 10000  # 10 second timeout

        # Verify identity
        idn = self.query("*IDN?")
        if "Siglent" not in idn and "SDG" not in idn.upper():
            print(f"Warning: Unexpected device ID: {idn}")

        print(f"Connected: {idn.strip()}")
        return True

    def disconnect(self):
        """Close connection."""
        if self.instr:
            self.instr.close()
            self.instr = None
        if self.rm:
            self.rm.close()
            self.rm = None

    def write(self, cmd: str):
        """Send SCPI command."""
        self.instr.write(cmd)

    def write_raw(self, data: bytes):
        """Send raw bytes (for binary waveform data)."""
        self.instr.write_raw(data)

    def query(self, cmd: str) -> str:
        """Send SCPI query and return response."""
        return self.instr.query(cmd)

    def reset(self):
        """Reset to default state."""
        self.write("*RST")
        self.write("*CLS")

    # --- Basic Waveform Control ---

    def set_function(self, func: Literal["SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB", "DC"]):
        """Set output function type."""
        self.write(self._cmd(f"BSWV WVTP,{func}"))

    def set_frequency(self, freq_hz: float):
        """Set output frequency in Hz."""
        self.write(self._cmd(f"BSWV FRQ,{freq_hz}"))

    def set_amplitude(self, vpp: float):
        """Set output amplitude in Vpp."""
        self.write(self._cmd(f"BSWV AMP,{vpp}"))

    def set_offset(self, volts: float):
        """Set DC offset in volts."""
        self.write(self._cmd(f"BSWV OFST,{volts}"))

    def set_output(self, enable: bool, load_hz: bool = True):
        """
        Enable or disable output.

        Args:
            enable: True to enable output
            load_hz: True for high-impedance load, False for 50 ohm
        """
        if enable:
            load = "HZ" if load_hz else "50"
            self.write(self._cmd(f"OUTP ON,LOAD,{load},PLTR,NOR"))
        else:
            self.write(self._cmd("OUTP OFF"))

    def get_output_state(self) -> bool:
        """Query output state."""
        response = self.query(self._cmd("OUTP?"))
        return "ON" in response.upper()

    # --- Burst Mode ---

    def set_burst_mode(self, enable: bool):
        """Enable/disable burst mode."""
        self.write(self._cmd(f"BTWV STATE,{'ON' if enable else 'OFF'}"))

    def set_burst_ncycles(self, cycles: int):
        """Set number of cycles per burst."""
        self.write(self._cmd(f"BTWV GATE_NCYC,NCYC"))
        self.write(self._cmd(f"BTWV TIME,{cycles}"))

    def set_burst_period(self, period_s: float):
        """Set burst period in seconds."""
        self.write(self._cmd(f"BTWV PRD,{period_s}"))

    def set_burst_trigger(self, source: Literal["INT", "EXT", "MAN"] = "MAN"):
        """Set burst trigger source."""
        self.write(self._cmd(f"BTWV TRSR,{source}"))

    def trigger(self):
        """Send manual trigger (for MAN trigger mode)."""
        self.write(self._cmd("BTWV MTRIG"))

    def configure_burst(self, freq_hz: float, vpp: float, ncycles: int,
                        trigger: Literal["INT", "EXT", "MAN"] = "MAN"):
        """Configure burst mode with specified parameters."""
        self.set_function("SINE")
        self.set_frequency(freq_hz)
        self.set_amplitude(vpp)
        self.set_burst_ncycles(ncycles)
        self.set_burst_trigger(trigger)
        self.set_burst_mode(True)

    # --- Arbitrary Waveform ---

    def upload_arbitrary(self, data: np.ndarray, name: str = "wave1") -> bool:
        """
        Upload arbitrary waveform data.

        Args:
            data: Waveform samples, will be normalized to [-1, +1]
            name: Waveform name for storage

        Returns:
            True if successful

        Notes:
            - SDG1000X DDS mode uses exactly 16384 points
            - Data is converted to 16-bit signed integers (-32768 to +32767)
            - TrueArb mode supports up to 8M points at variable sample rate
        """
        # Normalize to [-1, +1]
        data = np.asarray(data, dtype=np.float64)
        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            data = data / max_abs

        # Clamp to valid range
        data = np.clip(data, -1.0, 1.0)

        # Convert to 16-bit signed integers
        int_data = (data * 32767).astype(np.int16)

        # Bug workaround: Siglent treats 0x0a (newline) as terminator in binary data
        # Replace with 0x0b (should have minimal effect on waveform)
        binary_data = int_data.tobytes().replace(b'\x0a', b'\x0b')

        # Build command with binary data
        cmd = f"C{self.channel}:WVDT WVNM,{name},WAVEDATA,".encode() + binary_data + b'\n'

        self.write_raw(cmd)

        # Wait for operation to complete
        self.query("*OPC?")

        return True

    def select_arbitrary(self, name: str = "wave1"):
        """Select a stored arbitrary waveform."""
        self.write(self._cmd(f"ARWV NAME,{name}"))

    def set_sample_rate_mode(self, mode: Literal["DDS", "TARB"] = "DDS"):
        """
        Set arbitrary waveform sample rate mode.

        Args:
            mode: "DDS" for 16384-point mode, "TARB" for TrueArb variable length
        """
        self.write(self._cmd(f"SRATE MODE,{mode}"))

    def set_sample_rate(self, rate_hz: float):
        """Set sample rate for TrueArb mode."""
        self.write(self._cmd(f"SRATE VALUE,{rate_hz}"))

    def set_arbitrary_waveform(self, freq_hz: Optional[float] = None,
                               vpp: float = 1.0):
        """
        Configure output for arbitrary waveform.

        Args:
            freq_hz: Repetition frequency of the waveform
            vpp: Output amplitude in Vpp
        """
        # Set function to ARB
        self.set_function("ARB")

        # Set amplitude
        self.set_amplitude(vpp)

        # Set frequency if provided
        if freq_hz:
            self.set_frequency(freq_hz)

    # --- Convenience Methods ---

    def output_sine(self, freq_hz: float, vpp: float, enable: bool = True):
        """Quick setup for continuous sine wave."""
        self.set_function("SINE")
        self.set_frequency(freq_hz)
        self.set_amplitude(vpp)
        self.set_offset(0)
        self.set_burst_mode(False)
        self.set_output(enable)

    def output_burst_sine(self, freq_hz: float, vpp: float, ncycles: int,
                          enable: bool = True):
        """Quick setup for triggered burst of sine cycles."""
        self.configure_burst(freq_hz, vpp, ncycles, trigger="MAN")
        self.set_output(enable)

    def output_arbitrary(self, data: np.ndarray, freq_hz: float, vpp: float,
                         name: str = "wave1", enable: bool = True):
        """
        Quick setup for arbitrary waveform output.

        Args:
            data: Waveform samples (will be normalized)
            freq_hz: Repetition frequency of the waveform
            vpp: Output amplitude
            name: Waveform storage name
            enable: Enable output immediately
        """
        self.upload_arbitrary(data, name=name)
        self.select_arbitrary(name)
        self.set_arbitrary_waveform(freq_hz=freq_hz, vpp=vpp)
        self.set_burst_mode(False)
        self.set_output(enable)

    def status(self) -> dict:
        """Query current instrument status."""
        # Siglent returns status in a different format, parse BSWV response
        try:
            bswv = self.query(self._cmd("BSWV?"))
            outp = self.query(self._cmd("OUTP?"))
            return {
                "bswv_response": bswv.strip(),
                "output": "ON" in outp.upper(),
            }
        except Exception as e:
            return {"error": str(e)}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# --- Waveform Generation Utilities (same as Agilent version) ---

def generate_gaussian_pulse(n_points: int, center_freq_hz: float,
                            sample_rate_hz: float, sigma_cycles: float = 3.0) -> np.ndarray:
    """
    Generate a Gaussian-enveloped sinusoidal pulse.

    Args:
        n_points: Number of samples
        center_freq_hz: Carrier frequency
        sample_rate_hz: Sample rate
        sigma_cycles: Envelope width in cycles of carrier

    Returns:
        Normalized waveform array
    """
    t = np.arange(n_points) / sample_rate_hz
    t_center = t[n_points // 2]
    sigma_t = sigma_cycles / center_freq_hz

    envelope = np.exp(-0.5 * ((t - t_center) / sigma_t) ** 2)
    carrier = np.sin(2 * np.pi * center_freq_hz * t)

    return envelope * carrier


def generate_shaped_pulse(n_points: int, center_freq_hz: float,
                          sample_rate_hz: float, fourier_coeffs: np.ndarray) -> np.ndarray:
    """
    Generate a pulse with Fourier-parameterized envelope.

    Args:
        n_points: Number of samples
        center_freq_hz: Carrier frequency
        sample_rate_hz: Sample rate
        fourier_coeffs: Array of [a0, a1, b1, a2, b2, ...] coefficients

    Returns:
        Normalized waveform array
    """
    t = np.arange(n_points) / sample_rate_hz
    t_norm = t / t[-1]  # Normalize to [0, 1]

    # Build envelope from Fourier series
    envelope = np.ones(n_points) * fourier_coeffs[0]  # DC term

    n_harmonics = (len(fourier_coeffs) - 1) // 2
    for k in range(1, n_harmonics + 1):
        idx_a = 2 * k - 1
        idx_b = 2 * k
        if idx_a < len(fourier_coeffs):
            envelope += fourier_coeffs[idx_a] * np.cos(2 * np.pi * k * t_norm)
        if idx_b < len(fourier_coeffs):
            envelope += fourier_coeffs[idx_b] * np.sin(2 * np.pi * k * t_norm)

    # Ensure non-negative envelope
    envelope = np.maximum(envelope, 0)

    # Apply carrier
    carrier = np.sin(2 * np.pi * center_freq_hz * t)
    waveform = envelope * carrier

    # Normalize
    max_abs = np.max(np.abs(waveform))
    if max_abs > 0:
        waveform = waveform / max_abs

    return waveform


if __name__ == "__main__":
    # Quick test
    print("Siglent SDG1000X Test")
    print("=" * 40)

    try:
        with SiglentSDG1000X() as awg:
            print("\nStatus:", awg.status())

            # Test continuous sine
            print("\nOutputting 40kHz sine at 2Vpp...")
            awg.output_sine(40000, 2.0)
            print("Status:", awg.status())

            input("\nPress Enter to test arbitrary waveform...")

            # Test arbitrary waveform
            print("\nUploading Gaussian pulse...")
            pulse = generate_gaussian_pulse(
                n_points=16000,  # For 62.5 Hz at 1 MHz effective rate
                center_freq_hz=40000,
                sample_rate_hz=1e6,
                sigma_cycles=5
            )
            awg.output_arbitrary(pulse, freq_hz=62.5, vpp=5.0)
            print("Status:", awg.status())

            input("\nPress Enter to disable output and exit...")
            awg.set_output(False)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTo connect manually, find your device with:")
        print("  import pyvisa")
        print("  rm = pyvisa.ResourceManager()")
        print("  print(rm.list_resources())")
