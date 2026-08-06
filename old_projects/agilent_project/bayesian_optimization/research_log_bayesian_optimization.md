# Bayesian Optimization for Acoustic Pulse Shaping

**Date:** May 27, 2026

## Motivation

The iterative local tangent method (Jacobian-based gradient descent) for pulse shaping was unsuccessful. After 9 iterations (~45 minutes of runtime), the L2 error metric improved from 0.56 to 0.41, yet the actual FWHM *increased* from 612 μs to 1034 μs—a 69% degradation. The optimizer found that increasing pulse amplitude reduced L2 error more easily than narrowing the pulse, effectively gaming the objective function.

The core issue: the target waveform (a narrow Gaussian with σ = 2 cycles, corresponding to FWHM ≈ 118 μs) is physically unachievable. Impulse response characterization showed the piezo system has an intrinsic minimum FWHM of ~372 μs due to resonant ringdown. Asking the optimizer to match an impossible target led to pathological behavior.

## Approach: Bayesian Optimization with Gaussian Process Surrogate

Rather than estimating local gradients via expensive Jacobian computation, we build a *global* probabilistic model of the objective function using a Gaussian process (GP). The GP provides both a prediction $\mu(\theta)$ and uncertainty estimate $\sigma(\theta)$ at any point in parameter space, enabling principled exploration-exploitation tradeoffs.

### Objective Function

We optimize FWHM directly:

$$J(\theta) = \text{FWHM}(y_\theta) + \lambda \cdot \max(0, A_{\min} - A_\theta)^2$$

where $y_\theta$ is the measured acoustic envelope for parameters $\theta$, $A_\theta$ is the peak amplitude, and the second term penalizes solutions that drop below 70% of baseline amplitude.

### Parameterization

Instead of optimizing all $N = 4000$ waveform samples, we use a 10-dimensional parameterization:

$$f_\theta(t) = A \exp\left[-\frac{(t - t_c)^2}{2\sigma^2}\right] \cos(2\pi f_0 t + \phi) + \sum_{k=1}^{3} c_k^{\text{pre}} g_k^{\text{pre}}(t) + \sum_{k=1}^{5} c_k^{\text{post}} g_k^{\text{post}}(t)$$

where the compensation functions $g_k(t)$ are windowed sinusoids placed before and after the main pulse to allow pre-excitation and ringdown cancellation.

The parameter vector is:
$$\theta = (\sigma, \phi, c_1^{\text{pre}}, c_2^{\text{pre}}, c_3^{\text{pre}}, c_1^{\text{post}}, c_2^{\text{post}}, c_3^{\text{post}}, c_4^{\text{post}}, c_5^{\text{post}})^T \in \mathbb{R}^{10}$$

### GP Surrogate Model

The objective is modeled as a Gaussian process:

$$J(\theta) \sim \mathcal{GP}(0, k(\theta, \theta'))$$

with squared exponential kernel:

$$k(\theta, \theta') = \sigma_f^2 \exp\left(-\frac{\|\theta - \theta'\|^2}{2\ell^2}\right)$$

Given observations $\mathcal{D}_n = \{(\theta_i, y_i)\}_{i=1}^n$, the posterior at a new point $\theta_*$ is:

$$\mu_n(\theta_*) = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\sigma_n^2(\theta_*) = k(\theta_*, \theta_*) - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

### Acquisition Function

We use Expected Improvement (EI) to select the next evaluation point:

$$\alpha_{\text{EI}}(\theta) = (J_{\min} - \mu_n(\theta)) \Phi(Z) + \sigma_n(\theta) \phi(Z)$$

where $Z = (J_{\min} - \mu_n(\theta)) / \sigma_n(\theta)$, and $\Phi$, $\phi$ are the standard normal CDF and PDF.

This balances exploitation (low $\mu_n$) with exploration (high $\sigma_n$).

## Experimental Setup

- **AWG:** Agilent 33220A, 1 MSa/s, 4000-point waveforms (4 ms period, 250 Hz rep rate)
- **Microphone:** Dodotronic Ultramic 384K, 384 kHz sampling rate
- **Transducers:** 40 kHz piezoelectric array (hexagonal, 7 elements)
- **Bandpass filter:** 4th-order Butterworth, 38-42 kHz
- **Envelope extraction:** Hilbert transform

Optimization parameters:
- Initial random samples: 20
- GP-guided iterations: 80
- Total evaluations: 100
- Runtime: ~2.5 minutes

## Results

### Baseline (σ = 5.0 cycles, no compensation)
- FWHM: 643 μs
- Peak amplitude: 1.26 (normalized)

### Optimized
- FWHM: **399 μs** (38% improvement)
- Peak amplitude: 1.09 (86% of baseline, within tolerance)

Best parameters found:
- σ = 1.19 cycles (near minimum bound)
- φ = 3.54 rad
- Pre-compensation: [0.36, -0.03, -0.25]
- Post-compensation: [-0.43, -0.50, -0.50, -0.39, 0.42]

The optimizer discovered that:
1. **Narrow input pulses work best.** The optimal σ ≈ 1.2 cycles is near the lower bound, confirming that the piezo's intrinsic response dominates—shorter inputs simply reduce the additional broadening from the input envelope.

2. **Post-pulse compensation is critical.** The optimizer saturated several post-compensation coefficients at -0.50, indicating that aggressive anti-phase excitation after the main pulse helps cancel ringdown.

3. **Pre-pulse compensation provides marginal benefit.** The pre-compensation amplitudes are smaller and mixed-sign, suggesting the system is less sensitive to pre-excitation.

## Comparison to Jacobian Method

| Aspect | Local Tangent (Jacobian) | Bayesian Optimization |
|--------|--------------------------|----------------------|
| Parameters | 2700 (all samples) | 10 (physically motivated) |
| Measurements/iteration | 270 (Jacobian estimation) | 1 |
| Total measurements | ~2700 for 10 iterations | 100 |
| Runtime | ~50 minutes | ~2.5 minutes |
| Objective | L2 vs impossible target | Direct FWHM |
| Result | FWHM increased 69% | FWHM decreased 38% |

## Figures

- **waveform_comparison_overlay.png**: Superimposed baseline (blue) and optimized (red) waveforms showing both input and output
- **baseline_waveform_detail.png**: Detailed view of baseline input/output
- **best_waveform_detail.png**: Detailed view of optimized input/output
- **bayesian_optimization_results.png**: Convergence plot and optimization summary

## Next Steps

1. **Matched filter diagnostic**: Use the optimized pulse as a template for cross-correlation-based TOF measurement
2. **Repeatability study**: Characterize measurement-to-measurement variability
3. **Temperature sensitivity**: Test pulse shape stability across operating temperatures
4. **Extended optimization**: Try larger parameter space (more compensation terms) or alternative kernels (Matérn)

## Files

```
bayesian_optimizer.py          # Main optimization code
bayesian_optimizer_state.npz   # Saved results (best_params, all_fwhms, etc.)
waveform_comparison_data.npz   # Baseline vs optimized envelopes
```
