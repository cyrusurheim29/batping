"""
Simulated Tangent Optimization

A toy experiment to test and refine the iterative local tangent method
without hardware. Uses a simulated nonlinear system S that mimics
piezo transducer behavior (resonance + ringdown).

This allows rapid iteration on algorithm parameters before applying
to real hardware.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import time


@dataclass
class SimulationResult:
    """Results from simulated optimization."""
    f_initial: np.ndarray
    f_final: np.ndarray
    error_history: List[float]
    y_initial: np.ndarray
    y_final: np.ndarray
    y_target: np.ndarray
    iterations: int


class SimulatedSystem:
    """
    Simulated piezo-acoustic system S.

    Models:
    1. Bandpass filtering around resonance frequency (40kHz equivalent)
    2. Nonlinear saturation
    3. Ringdown (exponential decay after excitation)
    4. Propagation delay
    5. Additive noise
    """

    def __init__(
        self,
        n_input: int = 400,      # Input waveform samples
        n_output: int = 600,     # Output samples
        resonance_freq: float = 0.1,  # Normalized resonance (0-0.5)
        Q_factor: float = 20,    # Quality factor (higher = more ringdown)
        delay_samples: int = 50, # Propagation delay
        noise_std: float = 0.01, # Measurement noise
        nonlinearity: float = 0.1,  # Saturation strength
    ):
        self.n_input = n_input
        self.n_output = n_output
        self.resonance_freq = resonance_freq
        self.Q_factor = Q_factor
        self.delay_samples = delay_samples
        self.noise_std = noise_std
        self.nonlinearity = nonlinearity

        # Design resonant bandpass filter
        bandwidth = resonance_freq / Q_factor
        low = max(0.01, resonance_freq - bandwidth/2)
        high = min(0.49, resonance_freq + bandwidth/2)
        self.b, self.a = signal.butter(4, [low, high], btype='band')

        # Ringdown envelope (exponential decay)
        decay_const = Q_factor / (2 * np.pi * resonance_freq)
        t = np.arange(n_output)
        self.ringdown_envelope = np.exp(-t / decay_const)

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """
        Apply simulated system transformation.

        Args:
            f: Input waveform (n_input samples)

        Returns:
            y: Output waveform (n_output samples)
        """
        # Pad input for output length
        f_padded = np.zeros(self.n_output)
        f_padded[self.delay_samples:self.delay_samples + len(f)] = f[:min(len(f), self.n_output - self.delay_samples)]

        # Bandpass filter (resonance)
        y = signal.filtfilt(self.b, self.a, f_padded)

        # Nonlinear saturation: y -> tanh(y / (1 - nonlinearity * |y|))
        # This creates harmonic distortion and amplitude compression
        y = np.tanh(y / (1 + self.nonlinearity * np.abs(y)))

        # Apply ringdown envelope (models mechanical resonator behavior)
        # Find the peak and apply decay after it
        envelope = np.abs(signal.hilbert(y))
        peak_idx = np.argmax(envelope)

        # Multiply by decaying envelope after peak
        ringdown = np.ones(self.n_output)
        ringdown[peak_idx:] = np.exp(-(np.arange(self.n_output - peak_idx)) / (self.Q_factor * 2))
        y = y * ringdown

        # Add noise
        y = y + self.noise_std * np.random.randn(self.n_output)

        # Extract envelope (like we do with Hilbert in real system)
        envelope = np.abs(signal.hilbert(y))

        return envelope


class SimulatedOptimizer:
    """
    Tangent optimizer for simulated system.

    Same algorithm as full_waveform_optimizer but runs in simulation.
    """

    def __init__(
        self,
        system: SimulatedSystem,
        epsilon: float = 0.05,
        lambda_reg: float = 0.01,
        alpha: float = 0.3,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        target_sigma: int = 10,  # Target pulse width in samples
        adaptive_alpha: bool = True,
    ):
        self.system = system
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.target_sigma = target_sigma
        self.adaptive_alpha = adaptive_alpha

    def generate_adaptive_target(self, y: np.ndarray) -> np.ndarray:
        """Generate narrow Gaussian target centered at peak of y."""
        peak_idx = np.argmax(y)
        peak_amp = y[peak_idx]

        t = np.arange(len(y))
        target = peak_amp * np.exp(-0.5 * ((t - peak_idx) / self.target_sigma) ** 2)

        return target

    def estimate_jacobian(self, f: np.ndarray, y_current: np.ndarray) -> np.ndarray:
        """Estimate Jacobian by finite differences."""
        N = len(f)
        M = len(y_current)
        K = np.zeros((M, N))

        for j in range(N):
            f_perturbed = f.copy()
            f_perturbed[j] += self.epsilon
            y_perturbed = self.system(f_perturbed)
            K[:, j] = (y_perturbed - y_current) / self.epsilon

        return K

    def compute_correction(self, K: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Regularized least-squares correction."""
        N = K.shape[1]
        KtK = K.T @ K
        Kt_e = K.T @ error
        A = KtK + self.lambda_reg * np.eye(N)
        b = -Kt_e
        delta_f = np.linalg.solve(A, b)
        return delta_f

    def optimize(
        self,
        f_initial: np.ndarray,
        verbose: bool = True
    ) -> SimulationResult:
        """Run optimization."""
        f = f_initial.copy()
        error_history = []

        # Store initial output
        y_initial = self.system(f_initial)

        if verbose:
            print(f"Simulated Tangent Optimization")
            print(f"Input: {len(f)} samples, Output: {self.system.n_output} samples")
            print(f"Parameters: ε={self.epsilon}, λ={self.lambda_reg}, α={self.alpha}")
            print(f"Target width: {self.target_sigma} samples")
            print()

        alpha = self.alpha
        prev_error = None

        for n in range(self.max_iterations):
            # Measure
            y = self.system(f)

            # Adaptive target
            y_target = self.generate_adaptive_target(y)

            # Error
            error = y - y_target
            error_norm = np.sqrt(np.mean(error**2))
            error_history.append(error_norm)

            if verbose:
                print(f"Iter {n+1}: error={error_norm:.6f}, α={alpha:.3f}")

            # Convergence check
            if error_norm < self.tolerance:
                if verbose:
                    print("Converged!")
                break

            # Adaptive step size: reduce if error increased
            if self.adaptive_alpha and prev_error is not None:
                if error_norm > prev_error * 1.1:  # Error increased by >10%
                    alpha *= 0.7  # Reduce step size
                    if verbose:
                        print(f"  -> Reducing α to {alpha:.3f}")
                elif error_norm < prev_error * 0.95:  # Good progress
                    alpha = min(alpha * 1.1, self.alpha)  # Increase slightly

            prev_error = error_norm

            # Jacobian
            K = self.estimate_jacobian(f, y)

            # Correction
            delta_f = self.compute_correction(K, error)

            # Update
            f = f + alpha * delta_f

            # Clip to valid range
            f = np.clip(f, -1, 1)

        # Final output
        y_final = self.system(f)
        y_target_final = self.generate_adaptive_target(y_final)

        return SimulationResult(
            f_initial=f_initial,
            f_final=f,
            error_history=error_history,
            y_initial=y_initial,
            y_final=y_final,
            y_target=y_target_final,
            iterations=len(error_history)
        )


def create_gaussian_input(n_samples: int, center: float = 0.5, sigma: float = 0.1,
                          carrier_freq: float = 40) -> np.ndarray:
    """Create Gaussian pulse input.

    Args:
        n_samples: Number of samples
        center: Center position (0-1)
        sigma: Envelope width (fraction of total)
        carrier_freq: Cycles per waveform (default 40 to match 0.1 normalized resonance)
    """
    t = np.linspace(0, 1, n_samples)
    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    return envelope * carrier


def plot_results(result: SimulationResult, save_path: str = "simulation_results.png"):
    """Visualize optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error convergence
    axes[0, 0].semilogy(range(1, len(result.error_history) + 1), result.error_history, 'b.-')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Error (RMSE)')
    axes[0, 0].set_title('Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # Input waveforms
    axes[0, 1].plot(result.f_initial, 'b-', alpha=0.7, label='Initial')
    axes[0, 1].plot(result.f_final, 'r-', alpha=0.7, label='Optimized')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Input Waveforms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Output envelopes
    axes[1, 0].plot(result.y_initial, 'b-', alpha=0.7, label='Initial output')
    axes[1, 0].plot(result.y_final, 'r-', alpha=0.7, label='Optimized output')
    axes[1, 0].plot(result.y_target, 'g--', alpha=0.7, label='Target')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Envelope')
    axes[1, 0].set_title('Output Envelopes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Waveform difference
    diff = result.f_final - result.f_initial
    axes[1, 1].plot(diff, 'g-')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'Correction (max={np.max(np.abs(diff)):.3f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_parameter_sweep():
    """Test different algorithm parameters."""
    print("="*60)
    print("Parameter Sweep")
    print("="*60)

    system = SimulatedSystem(
        n_input=400,
        n_output=600,
        Q_factor=20,
        noise_std=0.01,
    )

    f_initial = create_gaussian_input(400, center=0.5, sigma=0.15)

    # Test different step sizes
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7]
    results = {}

    print("\nTesting step sizes:")
    for alpha in alphas:
        optimizer = SimulatedOptimizer(
            system=system,
            alpha=alpha,
            adaptive_alpha=False,
            max_iterations=30,
        )
        result = optimizer.optimize(f_initial, verbose=False)
        final_error = result.error_history[-1]
        best_error = min(result.error_history)
        results[alpha] = (final_error, best_error, result.error_history)
        print(f"  α={alpha}: final={final_error:.4f}, best={best_error:.4f}")

    # Test adaptive vs fixed
    print("\nAdaptive step size:")
    optimizer = SimulatedOptimizer(
        system=system,
        alpha=0.5,
        adaptive_alpha=True,
        max_iterations=30,
    )
    result = optimizer.optimize(f_initial, verbose=False)
    print(f"  Adaptive: final={result.error_history[-1]:.4f}, best={min(result.error_history):.4f}")

    return results


def test_fixed_vs_adaptive_target():
    """Compare fixed target vs adaptive target approaches."""
    print("\n" + "="*60)
    print("Fixed vs Adaptive Target Comparison")
    print("="*60)

    system = SimulatedSystem(
        n_input=400,
        n_output=600,
        Q_factor=20,
        noise_std=0.005,
    )

    f_initial = create_gaussian_input(400, center=0.5, sigma=0.15)

    # First get initial output to set fixed target
    y_init = system(f_initial)
    peak_idx = np.argmax(y_init)
    peak_amp = y_init[peak_idx]

    # Fixed target: same Gaussian but doesn't move
    def generate_fixed_target(y):
        t = np.arange(len(y))
        return peak_amp * np.exp(-0.5 * ((t - peak_idx) / 10) ** 2)

    class FixedTargetOptimizer(SimulatedOptimizer):
        def generate_adaptive_target(self, y):
            return generate_fixed_target(y)

    # Run with fixed target
    print("\nFixed target (center locked at initial peak):")
    fixed_opt = FixedTargetOptimizer(
        system=system,
        alpha=0.3,
        max_iterations=50,
        target_sigma=10,
    )
    fixed_result = fixed_opt.optimize(f_initial, verbose=False)
    print(f"  Initial: {fixed_result.error_history[0]:.4f}")
    print(f"  Final: {fixed_result.error_history[-1]:.4f}")
    print(f"  Best: {min(fixed_result.error_history):.4f}")
    print(f"  Improvement: {(1 - min(fixed_result.error_history)/fixed_result.error_history[0])*100:.1f}%")

    # Run with adaptive target
    print("\nAdaptive target (center tracks peak):")
    adaptive_opt = SimulatedOptimizer(
        system=system,
        alpha=0.3,
        max_iterations=50,
        target_sigma=10,
    )
    adaptive_result = adaptive_opt.optimize(f_initial, verbose=False)
    print(f"  Initial: {adaptive_result.error_history[0]:.4f}")
    print(f"  Final: {adaptive_result.error_history[-1]:.4f}")
    print(f"  Best: {min(adaptive_result.error_history):.4f}")
    print(f"  Improvement: {(1 - min(adaptive_result.error_history)/adaptive_result.error_history[0])*100:.1f}%")

    # Test with lower Q (easier system)
    print("\n" + "-"*40)
    print("Testing with lower Q factor (less ringdown):")
    easy_system = SimulatedSystem(
        n_input=400,
        n_output=600,
        Q_factor=5,  # Much lower Q
        noise_std=0.005,
    )

    adaptive_easy = SimulatedOptimizer(
        system=easy_system,
        alpha=0.3,
        max_iterations=50,
        target_sigma=10,
    )
    easy_result = adaptive_easy.optimize(f_initial, verbose=False)
    print(f"  Initial: {easy_result.error_history[0]:.4f}")
    print(f"  Final: {easy_result.error_history[-1]:.4f}")
    print(f"  Best: {min(easy_result.error_history):.4f}")
    print(f"  Improvement: {(1 - min(easy_result.error_history)/easy_result.error_history[0])*100:.1f}%")


def main():
    """Run simulation experiment."""
    print("="*60)
    print("Simulated Tangent Optimization Experiment")
    print("="*60)

    # Create system that mimics piezo behavior
    system = SimulatedSystem(
        n_input=400,
        n_output=600,
        resonance_freq=0.1,
        Q_factor=20,        # High Q = lots of ringdown
        delay_samples=50,
        noise_std=0.005,
        nonlinearity=0.1,
    )

    # Initial waveform (Gaussian pulse)
    f_initial = create_gaussian_input(400, center=0.5, sigma=0.15)

    print(f"\nSystem parameters:")
    print(f"  Q factor: {system.Q_factor}")
    print(f"  Noise std: {system.noise_std}")
    print(f"  Nonlinearity: {system.nonlinearity}")
    print()

    # Run optimization with adaptive step size
    optimizer = SimulatedOptimizer(
        system=system,
        epsilon=0.05,
        lambda_reg=0.01,
        alpha=0.5,
        adaptive_alpha=True,
        max_iterations=50,
        target_sigma=8,
    )

    t0 = time.time()
    result = optimizer.optimize(f_initial, verbose=True)
    elapsed = time.time() - t0

    print(f"\nCompleted {result.iterations} iterations in {elapsed:.1f}s")
    print(f"Initial error: {result.error_history[0]:.4f}")
    print(f"Final error: {result.error_history[-1]:.4f}")
    print(f"Best error: {min(result.error_history):.4f}")
    print(f"Improvement: {(1 - min(result.error_history)/result.error_history[0])*100:.1f}%")

    # Plot
    plot_results(result, "simulation_results.png")

    # Parameter sweep
    print()
    run_parameter_sweep()

    # Compare fixed vs adaptive target
    test_fixed_vs_adaptive_target()


if __name__ == "__main__":
    main()
