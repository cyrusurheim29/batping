"""
Agilent 33220A Function Generator Interface

Provides USB/SCPI control for the 33220A including arbitrary waveform upload.
"""

import numpy as np
import pyvisa
from typing import Optional, Literal


class Agilent33220A:
    """Interface to Agilent 33220A 20MHz Function Generator."""

    VENDOR_ID = 0x0957
    PRODUCT_ID = 0x0407

    def __init__(self):
        self.rm: Optional[pyvisa.ResourceManager] = None
        self.instr: Optional[pyvisa.resources.Resource] = None
        self.resource_name: Optional[str] = None

    def connect(self, resource_name: Optional[str] = None) -> bool:
        """Connect to the 33220A. Auto-detects if resource_name not provided."""
        self.rm = pyvisa.ResourceManager()

        if resource_name:
            self.resource_name = resource_name
        else:
            # Auto-detect by vendor/product ID
            resources = self.rm.list_resources()
            for res in resources:
                if f'{self.VENDOR_ID:04X}' in res.upper() or f'{self.VENDOR_ID:#06x}'[2:] in res:
                    if f'{self.PRODUCT_ID:04X}' in res.upper() or f'{self.PRODUCT_ID:#06x}'[2:] in res:
                        self.resource_name = res
                        break

            if not self.resource_name:
                raise RuntimeError(f"Agilent 33220A not found. Available: {resources}")

        self.instr = self.rm.open_resource(self.resource_name)
        self.instr.timeout = 10000  # 10 second timeout

        # Verify identity
        idn = self.query("*IDN?")
        if "33220A" not in idn:
            raise RuntimeError(f"Unexpected device: {idn}")

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

    def query(self, cmd: str) -> str:
        """Send SCPI query and return response."""
        return self.instr.query(cmd)

    def reset(self):
        """Reset to default state."""
        self.write("*RST")
        self.write("*CLS")

    # --- Basic Waveform Control ---

    def set_function(self, func: Literal["SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "USER"]):
        """Set output function type."""
        self.write(f"FUNC {func}")

    def set_frequency(self, freq_hz: float):
        """Set output frequency in Hz."""
        self.write(f"FREQ {freq_hz}")

    def set_amplitude(self, vpp: float):
        """Set output amplitude in Vpp."""
        self.write(f"VOLT {vpp}")

    def set_offset(self, volts: float):
        """Set DC offset in volts."""
        self.write(f"VOLT:OFFS {volts}")

    def set_output(self, enable: bool):
        """Enable or disable output."""
        self.write(f"OUTP {'ON' if enable else 'OFF'}")

    def get_output_state(self) -> bool:
        """Query output state."""
        return self.query("OUTP?").strip() == "1"

    # --- Burst Mode ---

    def set_burst_mode(self, enable: bool):
        """Enable/disable burst mode."""
        self.write(f"BURS:STAT {'ON' if enable else 'OFF'}")

    def set_burst_ncycles(self, cycles: int):
        """Set number of cycles per burst."""
        self.write(f"BURS:NCYC {cycles}")

    def set_burst_trigger(self, source: Literal["IMM", "EXT", "BUS"] = "BUS"):
        """Set burst trigger source."""
        self.write(f"TRIG:SOUR {source}")

    def trigger(self):
        """Send software trigger (for BUS trigger mode)."""
        self.write("*TRG")

    def configure_burst(self, freq_hz: float, vpp: float, ncycles: int,
                        trigger: Literal["IMM", "EXT", "BUS"] = "BUS"):
        """Configure burst mode with specified parameters."""
        self.set_function("SIN")
        self.set_frequency(freq_hz)
        self.set_amplitude(vpp)
        self.set_burst_ncycles(ncycles)
        self.set_burst_trigger(trigger)
        self.set_burst_mode(True)

    # --- Arbitrary Waveform ---

    def upload_arbitrary(self, data: np.ndarray, name: str = "VOLATILE") -> bool:
        """
        Upload arbitrary waveform data.

        Args:
            data: Waveform samples, will be normalized to [-1, +1]
            name: Waveform name (use "VOLATILE" for temporary storage)

        Returns:
            True if successful

        Notes:
            - 33220A accepts 1 to 16384 points for volatile memory
            - Values must be in range -1.0 to +1.0
            - Sample rate is determined by frequency setting and point count
        """
        # Normalize to [-1, +1]
        data = np.asarray(data, dtype=np.float64)
        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            data = data / max_abs

        # Clamp to valid range
        data = np.clip(data, -1.0, 1.0)

        # Check point count (33220A volatile memory: 1-16384 points)
        if len(data) < 1 or len(data) > 16384:
            raise ValueError(f"Data length {len(data)} must be 1-16384 points")

        # Format as comma-separated values
        data_str = ",".join(f"{v:.6f}" for v in data)

        # Upload to volatile memory
        self.write(f"DATA VOLATILE,{data_str}")

        # Wait for operation to complete
        self.query("*OPC?")

        return True

    def set_arbitrary_waveform(self, sample_rate_hz: Optional[float] = None,
                               freq_hz: Optional[float] = None,
                               vpp: float = 1.0):
        """
        Configure output for arbitrary waveform.

        Either sample_rate_hz or freq_hz should be specified:
        - freq_hz: The waveform repeats at this frequency
        - sample_rate_hz: Effective sample rate = freq_hz * num_points

        Args:
            sample_rate_hz: Desired sample rate (calculates freq from point count)
            freq_hz: Desired repetition frequency
            vpp: Output amplitude in Vpp
        """
        # Select volatile waveform as active user waveform
        self.write("FUNC:USER VOLATILE")

        # Set function to USER (arbitrary)
        self.write("FUNC USER")

        # Set amplitude
        self.set_amplitude(vpp)

        # Set frequency
        if freq_hz:
            self.set_frequency(freq_hz)
        elif sample_rate_hz:
            # Query number of points in current volatile waveform
            # Note: 33220A doesn't have a direct query for this,
            # so caller should track point count
            print("Warning: sample_rate_hz requires knowing point count")

    # --- Convenience Methods ---

    def output_sine(self, freq_hz: float, vpp: float, enable: bool = True):
        """Quick setup for continuous sine wave."""
        self.set_function("SIN")
        self.set_frequency(freq_hz)
        self.set_amplitude(vpp)
        self.set_offset(0)
        self.set_burst_mode(False)
        self.set_output(enable)

    def output_burst_sine(self, freq_hz: float, vpp: float, ncycles: int,
                          enable: bool = True):
        """Quick setup for triggered burst of sine cycles."""
        self.configure_burst(freq_hz, vpp, ncycles, trigger="BUS")
        self.set_output(enable)

    def output_arbitrary(self, data: np.ndarray, freq_hz: float, vpp: float,
                         enable: bool = True):
        """
        Quick setup for arbitrary waveform output.

        Args:
            data: Waveform samples (will be normalized)
            freq_hz: Repetition frequency of the waveform
            vpp: Output amplitude
            enable: Enable output immediately
        """
        self.upload_arbitrary(data)
        self.set_arbitrary_waveform(freq_hz=freq_hz, vpp=vpp)
        self.set_burst_mode(False)
        self.set_output(enable)

    def status(self) -> dict:
        """Query current instrument status."""
        return {
            "function": self.query("FUNC?").strip(),
            "frequency": float(self.query("FREQ?").strip()),
            "amplitude": float(self.query("VOLT?").strip()),
            "offset": float(self.query("VOLT:OFFS?").strip()),
            "output": self.get_output_state(),
            "burst": self.query("BURS:STAT?").strip() == "1",
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# --- Waveform Generation Utilities ---

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


def generate_compact_pulse(n_points: int, center_freq_hz: float,
                           sample_rate_hz: float, fourier_coeffs: np.ndarray,
                           active_fraction: float = 0.25,
                           center_position: float = 0.5) -> np.ndarray:
    """
    Generate a compact pulse using Fourier envelope over a fraction of the waveform.
    The rest of the waveform is zero (silent).

    Args:
        n_points: Total number of samples
        center_freq_hz: Carrier frequency
        sample_rate_hz: Sample rate
        fourier_coeffs: Array of [a0, a1, b1, a2, b2, ...] coefficients
        active_fraction: Fraction of waveform that is active (0.0 to 1.0)
        center_position: Where to center the active region (0.0 to 1.0)

    Returns:
        Normalized waveform array with compact pulse
    """
    t = np.arange(n_points) / sample_rate_hz

    # Calculate active region bounds
    active_samples = int(n_points * active_fraction)
    center_sample = int(n_points * center_position)
    start_sample = max(0, center_sample - active_samples // 2)
    end_sample = min(n_points, start_sample + active_samples)
    start_sample = max(0, end_sample - active_samples)  # Adjust if hit end

    # Create envelope (zero everywhere except active region)
    envelope = np.zeros(n_points)

    # Build Fourier envelope only in active region
    active_t = np.linspace(0, 1, end_sample - start_sample)

    active_envelope = np.ones(len(active_t)) * fourier_coeffs[0]
    n_harmonics = (len(fourier_coeffs) - 1) // 2
    for k in range(1, n_harmonics + 1):
        idx_a = 2 * k - 1
        idx_b = 2 * k
        if idx_a < len(fourier_coeffs):
            active_envelope += fourier_coeffs[idx_a] * np.cos(2 * np.pi * k * active_t)
        if idx_b < len(fourier_coeffs):
            active_envelope += fourier_coeffs[idx_b] * np.sin(2 * np.pi * k * active_t)

    # Apply smooth window to avoid discontinuities at edges
    # Use a Tukey window (tapered cosine)
    from scipy.signal import windows
    taper = windows.tukey(len(active_t), alpha=0.3)
    active_envelope = active_envelope * taper

    # Ensure non-negative
    active_envelope = np.maximum(active_envelope, 0)

    envelope[start_sample:end_sample] = active_envelope

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
    with Agilent33220A() as awg:
        print("\nStatus:", awg.status())

        # Test continuous sine
        print("\nOutputting 40kHz sine at 2Vpp...")
        awg.output_sine(40000, 2.0)
        print("Status:", awg.status())

        input("\nPress Enter to test arbitrary waveform...")

        # Test arbitrary waveform
        print("\nUploading Gaussian pulse...")
        pulse = generate_gaussian_pulse(
            n_points=4000,
            center_freq_hz=40000,
            sample_rate_hz=1e6,
            sigma_cycles=5
        )
        awg.output_arbitrary(pulse, freq_hz=250, vpp=2.0)  # 250 Hz rep rate = 4ms period
        print("Status:", awg.status())

        input("\nPress Enter to disable output and exit...")
        awg.set_output(False)
