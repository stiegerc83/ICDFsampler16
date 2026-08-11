import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Ensure the script can locate your compiled module if you are running it inside a subdirectory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import icdf_sampler_16
except ImportError:
    print("Error: Could not import 'icdf_sampler_16'. Ensure you ran 'maturin develop --features python'.")
    sys.exit(1)

def main():
    (xmin, xmax) = (-3, 3)
    num_pts = 10_000
    
    x = np.linspace(xmin, xmax, num_pts, dtype=np.float64)
    # The analytical CDF of 1/sqrt(pi) * exp(-x^2) is 0.5 * (1 + erf(x)).
    cdf = 0.5 * (1.0 + erf(x))

    # Initialize your Rust engine via the bridge wrapper.
    sampler = icdf_sampler_16.ICDFSampler16(x.tolist(), cdf.tolist())

    # Pull all 65,536 (2^16) deterministic samples back out instantly.
    samples = np.array(sampler.sample_all(), dtype=np.float32)

    # Calculate the true theoretical probability for each generated sample value
    empirical_u = np.linspace(0.0, 1.0, len(samples))
    theoretical_u = 0.5 * (1.0 + erf(samples))

    # The error represents the discrepancy in probability mapping (quantization/interpolation drift)
    interpolation_error = empirical_u - theoretical_u

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reconstructed Distribution Histogram profile
    ax1.hist(
        samples,
        bins=100,
        density=True,
        alpha=0.6,
        color="royalblue",
        label="Rust Sampler Output",
    )
    # Overlay theoretical PDF curve
    x_plot = np.linspace(xmin - 0.5, xmax + 0.5, 500)
    pdf_theoretical = (1.0 / np.sqrt(np.pi)) * np.exp(-(x_plot**2))
    ax1.plot(
        x_plot,
        pdf_theoretical,
        color="crimson",
        linewidth=2,
        label="Theoretical PDF",
    )
    ax1.set_title("Reconstructed Population Shape")
    ax1.set_xlabel("Value (x)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Exact Interpolation Error Profile
    ax2.plot(
        samples,
        interpolation_error,
        color="darkorange",
        linewidth=1,
        label="CDF Error ($\Delta U$)",
    )
    ax2.set_title("Quantization & Interpolation Error Profile")
    ax2.set_xlabel("Sample Value (x)")
    ax2.set_ylabel("Error Matrix (Empirical U - Theoretical U)")
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Calculate statistics and display them on the plot window title
    empirical_variance = np.var(samples)
    print(f"--- Verification Statistics ---")
    print(f"Empirical Mean:     {np.mean(samples):.6f}")
    print(f"Empirical Variance: {empirical_variance:.6f} (Theoretical Target: 0.5)")
    print(f"Max Absolute Error: {np.max(np.abs(interpolation_error)):.6e}")

    plt.suptitle(
        f"ICDFSampler16 Diagnostics — Variance: {empirical_variance:.5f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
