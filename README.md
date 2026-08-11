# Cache-Pinned Adaptive ICDF Sampler

A zero-dependency, pure Rust Inverse Cumulative Distribution Function (ICDF) sampler designed for high-performance simulations, such as Ising model thermal loops. It compresses continuous probability distributions down to an optimized, branchless fixed-point layout that executes inside single-digit CPU clock cycles.

## What It Is

The library takes an arbitrary continuous distribution, calculates its true empirical L1 interpolation errors, and runs an offline Dynamic Programming knapsack solver to find the mathematically optimal point distribution.

The resulting runtime architecture is completely frozen and optimized for hardware efficiency:
* **The Budget**: Exactly 257 total data points are distributed across 32 spatial bins.
* **The Probe**: A raw `u16` integer acts as a unified fixed-point fraction over the closed interval.
* **The Critical Path**: Completely branchless and division-free. It uses a 64-byte index table designed to fit entirely within a single L1 CPU cache line for deterministic 1-cycle lookups, resolving samples via hardware bit-shifts and Fused Multiply-Adds (`mul_add`).

## Architectural Boundaries

This sampler is a highly specialized tool designed for a specific precision sweet spot, not an infinitely scalable generic container:
* **Stiff Range Boundaries**: It maps the data it is given from the absolute lowest x to the highest x (0 maps to the lowest value, 65535 maps to the highest value).
* **Tail Limitations**: If your simulation requires extreme, multi-sigma freak events far out in the exponential tails, expanding the range inside this fixed budget will starve the resolution of the bulk distribution where the meat of the data sits. Capturing chaotic extremes requires either wide intervals at the cost of core precision, or a completely different analytical tail-handling approach.
* **Scale Limits**: This design is strictly bounded by a 16-bit probe and a `u8` internal indexing structure. Attempting to scale to `u32` inputs or thousands of points will break the single-cache-line invariant and require a complete rewrite using wider integer types.

## Technical Documentation

Every single micro-architectural detail, register bitmask layout, and dynamic programming state-transition matrix is explained at length via verbose doc-comments directly inside the source code (`src/lib.rs`). Run `cargo doc --open` to view the full structural breakdown.
