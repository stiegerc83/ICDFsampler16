/// Generates a vector of `count` of linear samples inside the interval `[xmin, xmax]`;
/// edges included.
/// Ranges with `xmin` >= `xmax` are allowed. Very large numbers might lead to problems.
pub fn linspace(xmin: f64, xmax: f64, count: usize) -> Vec<f64> {
    match count {
        0 => vec![],
        1 => vec![xmin],
        10_000_000.. => panic!("you can't be serious!"),
        2.. => {
            let mut space = Vec::with_capacity(count);
            space.push(xmin); // Ensures exact binary clone of the input data.

            // Linear interpolation between the edges.
            let xrange = xmax - xmin;
            let scaler = 1.0 / (count - 1) as f64;
            for i in 1..(count - 1) {
                space.push(xmin + xrange * (i as f64 * scaler));
            }

            space.push(xmax); // Ensures exact binary clone of the input data.

            space
        }
    }
}

/// Errors possible during handling distributions.
#[derive(Debug, Eq, PartialEq)]
pub enum DistributionError {
    /// `x` and `f` are of unequal length.
    LenMismatch,
    /// The distribution as less than 2 data points.
    Underpopulated,
    /// The cumulative distribution is not strictly monotonically increasing.
    NotMonotonic,
}

/// Calculates the numerical intergration of f(x) using the trapez rule.
/// Returns an error for `x`, `f` of unequal length.
pub fn trapz(x: &[f64], f: &[f64]) -> Result<f64, DistributionError> {
    if x.len() != f.len() {
        return Err(DistributionError::LenMismatch);
    }

    let mut area = 0.0;

    let tuple_iter = x.iter().zip(f.iter());
    for ((x0, f0), (x1, f1)) in tuple_iter.clone().zip(tuple_iter.skip(1)) {
        area += 0.5 * (f1 + f0) * (x1 - x0);
    }

    Ok(area)
}

/// Calculates the numerical integration of f(x) for every x along the
/// integration axis. See `trapz`.
pub fn cumtrapz(x: &[f64], f: &[f64]) -> Result<Vec<f64>, DistributionError> {
    if x.len() != f.len() {
        return Err(DistributionError::LenMismatch);
    }

    let mut area = Vec::with_capacity(std::cmp::min(x.len(), f.len()));
    area.push(0.0);

    let tuple_iter = x.iter().zip(f.iter());
    for ((x0, f0), (x1, f1)) in tuple_iter.clone().zip(tuple_iter.skip(1)) {
        area.push(area.last().unwrap() + 0.5 * (f1 + f0) * (x1 - x0));
    }

    Ok(area)
}

/// Computes the Cumulative Distribution Function (`cdf`) from the distribution `f`.
/// Returns an error if
/// - the input is of unequal length.
/// - the input has less than 2 data points.
pub fn cdf_from_distribution(x: &[f64], f: &[f64]) -> Result<Vec<f64>, DistributionError> {
    let mut cdf = cumtrapz(x, f)?;
    match cdf.as_slice() {
        // Even a linear distribution needs at least 2 points.
        [] | [_] => Err(DistributionError::Underpopulated),

        [first, .., total_area] => {
            let total_area = *total_area;
            let mut prev_segment = *first;
            // Check the `cdf` is strictly monotonically increasing
            // and normalize it such that it spans exactly from 0.0
            // to 1.0.
            for segment in cdf.iter_mut().skip(1) {
                if *segment <= prev_segment {
                    return Err(DistributionError::NotMonotonic);
                }
                prev_segment = *segment;
                *segment /= total_area;
            }
            Ok(cdf)
        }
    }
}

/// Sampler for an Inverse Cumulative Distribution Function (ICDF; F^-1(u)).
/// Maps a `u16` directly to u in [0, 1] for sampling F^-1(u).
#[derive(Clone, Debug)]
pub struct ICDFSampler16 {
    /// 32 consecutive `[start_idx, interp_bits]: [u8; 2]` i.e. 64 bytes of metadata
    /// fitting exactly into one cache line which is 64 bytes on most modern CPU's
    /// (the OS has no control over it).
    ///
    /// `start_idx` is the index where the data points for a specific bin start
    /// inside `vals`; `6 <= interp_bits <= 11` is the number of bits used for linear
    /// interpolation. Thus the number of bits used for indexing inside one bin
    /// is `indexing_bits = 11 - interp_bits`.
    indexing: [u8; 64],

    /// Data points sitting on the inverse cumulative distribution F^-1(u) that is
    /// being sampled with a (random) `u16` intepreted as a uniform grid of 2^16
    /// points subdividing the interval u = [0, 1].
    ///
    /// Which points belong to which of the 32 bins is defined by `indexing`.
    /// This happens entirely using `u8` for all but the last point at index 256
    /// which is a padding to close off the last bin.
    ///
    /// This is needed because bins in a series |_|_|...|_| always need one more
    /// wall | to close off each bin on both sides.
    vals: [f32; 257],
}

impl ICDFSampler16 {
    /// 32 bins with 32 points each + 1 for padding the last bin.
    const MAX_GRID_SIZE: usize = 32 * 32 + 1;

    /// Stability parameter for error functions.
    const EPS: f32 = 1e-4;

    /// Creates a new sampler on the basis of a cumulative distribution function
    /// given as `(x, cdf)`; using a custom `err_fn` to generate the weights
    /// during the automatic determination of the ideal number of data points
    /// per bin.
    ///
    /// Returns `(Self, total_err)`; `total_err` is the total error due to thinning
    /// out points from the full grid according to `err_fn`.
    pub fn new_with_error_fn<F>(x: &[f64], cdf: &[f64], err_fn: F) -> (Self, f32)
    where
        F: Fn(f32, f32) -> f32,
    {
        // Get the inverse cumulative distribution function resampled on a uniform grid.
        let cdf_inv = Self::get_cdf_inv(x, cdf);

        // Get weights for each bin and allowed number of points using the error function
        // |f_interp - f_ref| at each interpolated point. These errors are summed up to
        // for a weight.
        let weights = Self::get_error_weights(&cdf_inv, err_fn);

        // Get the optimized number of points in each bin
        let (bin_npts, total_err) = Self::get_bin_npts(&weights);

        // Create the indexing as 32 consecutive `[u8; 2]` alongside
        // `vals`, the array of 257 corresponding points copied from `cdf_inv`.
        let mut indexing = [0_u8; 64];
        let mut vals = [0.0_f32; 257];
        let mut idx = 0;
        for (bin_idx, pow) in bin_npts.into_iter().enumerate() {
            indexing[bin_idx * 2] = idx.try_into().unwrap();
            indexing[bin_idx * 2 + 1] = 11 - pow;

            // The step size depending on the number of points, i.e. 32 points -> step 1.
            let step = 1 << (5 - pow);
            let bin_slice = &cdf_inv[(bin_idx * 32)..((bin_idx + 1) * 32)];
            for val in bin_slice.iter().step_by(step) {
                vals[idx as usize] = *val;
                idx += 1;
            }
        }
        // The padding is always the last point of `cdf_inv` since it closes off the last bin.
        vals[256] = cdf_inv[1024];

        (Self { indexing, vals }, total_err)
    }

    /// Creates a new sampler using the symmetrix mean relative error function:
    /// |f_interp - f_ref| / (1/2 * (|f_interp| + |f_ref|) + eps).
    ///
    /// See `new_with_error_fn`.
    pub fn from_symmetric_mean_relative_errors(x: &[f64], cdf: &[f64]) -> (Self, f32) {
        Self::new_with_error_fn(x, cdf, |f_interp, f_ref| {
            (f_interp - f_ref).abs() / (0.5 * (f_interp.abs() + f_ref.abs()) + Self::EPS)
        })
    }

    /// Creates a new sampler using the symmetrix max relative error function:
    /// |f_interp - f_ref| / (max(|f_interp|, |f_ref|) + eps).
    /// This function is more robust for errors with very small numbers compared
    /// to the symmetric mean relative error function.
    ///
    /// See `new_with_error_fn`.
    pub fn from_symmetric_max_relative_errors(x: &[f64], cdf: &[f64]) -> (Self, f32) {
        Self::new_with_error_fn(x, cdf, |f_interp, f_ref| {
            (f_interp - f_ref).abs() / (f_interp.abs().max(f_ref.abs()) + Self::EPS)
        })
    }

    /// Computes the inverse Cumulative Distribution Function F^-1(u) (`cdf_inv`)
    /// by resampling the F(x) (`cdf`) on an even grid using 32 * 32 + 1 data points.
    ///
    /// This function may panic or produce nonsensical results when basic assumptions break:
    /// - Unequal number of data points in `x` vs `cdf`.
    /// - Not enough data points in `x` or `f`.
    /// - `cdf` is not strictly monotonically increasing.
    /// - special floating point values are present in `x` or `cdf`.
    fn get_cdf_inv(x: &[f64], cdf: &[f64]) -> [f32; Self::MAX_GRID_SIZE] {
        // Compute F^-1(u) (`cdf_inv`) by resampling in `cdf` to find the corresponding
        // value in `x` using linear interpolation.
        let u = match cdf {
            [first, .., last] => linspace(*first, *last, Self::MAX_GRID_SIZE),
            _ => unreachable!("bad cdf"),
        };

        let mut cdf_inv = [0.0_f32; Self::MAX_GRID_SIZE];
        for (i, s) in u.iter().enumerate() {
            match cdf.binary_search_by(|p| p.partial_cmp(s).unwrap()) {
                Ok(idx) => {
                    cdf_inv[i] = x[idx] as f32;
                }
                Err(idx) => {
                    let (f0, f1) = (cdf[idx - 1], cdf[idx]);
                    let (x0, x1) = (x[idx - 1], x[idx]);
                    let x_interp = x0 + (x1 - x0) / (f1 - f0) * (s - f0);
                    cdf_inv[i] = x_interp as f32;
                }
            }
        }

        cdf_inv
    }

    /// Computes the sum of errors as defined by the `err_fn` (e.g. `|f_interp - f_ref|`)
    /// for each number of data points in each bin.
    /// For the full set of points (32) this is assumed to be 0, for 16, 8, 4, 2, 1 points
    /// the linear interpolation is used for the thinned out points and used to compute the
    /// total error for this specific bin.
    ///
    /// Returns a matrix 32 (bin0, bin1, ..) by 6 (1, 2, .. 32) of weights, i.e. the index
    /// at which a weight is found corresponds to the power of 2 exponent of the number of
    /// points, at 0 -> 2^0 = 1, .., at 5 -> 2^5 = 32.
    fn get_error_weights<F>(cdf_inv: &[f32; Self::MAX_GRID_SIZE], err_fn: F) -> [[f32; 6]; 32]
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut weights = [[0.0; 6]; 32];
        for (bin_idx, weights) in weights.iter_mut().enumerate() {
            // The slice holding the points inside one specific bin at `bin_idx`
            // and the first point of the next bin or the padding for the last bin.
            let bin_slice = &cdf_inv[(bin_idx * 32)..=((bin_idx + 1) * 32)];

            // Iterate over the possible number of points a bin can hold,
            // i.e. 1(32), 2(16), 4(8), 8(4), 16(2) or 32(1) with the step
            // in parentheses. Note: 32 has no error, so we stop at 16 points.
            for (pts_idx, weight) in weights.iter_mut().enumerate() {
                let step = 1 << (5 - pts_idx);
                for w in bin_slice.windows(step + 1).step_by(step) {
                    match w {
                        [f0, interp @ .., f1] => {
                            let s = (*f1 - *f0) / step as f32;
                            let err = (1..step)
                                .map(|t| f0 + s * t as f32)
                                .zip(interp.iter())
                                .fold(0.0, |acc, (interp, refp)| acc + err_fn(interp, *refp));
                            *weight += err;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        weights
    }

    /// Computes the ideal number of points in each of the 32 bins on the basis of
    /// `weights` (see `get_error_weights`). The total budget for summed up steps
    /// is 256.
    ///
    /// The algorithm uses a dynamic programming approach where intermediate total
    /// errors for sums of number of points are stored on a table such that the final
    /// sequence can be retrieved through backtracking from the last cell.
    ///
    /// budget               1   2   3   4   5   6   7   8   9   ...  256
    /// ------------------------------------------------------------------
    /// bin0               | e | e | x | e | x | x | x | e | e | ... | x |
    /// bin0 + bin1        | x | e | e | e | e | e | x | e | e | ... | x |
    /// bin0 + bin1 + bin2 | x | x | e | e | e | e | e | e | e | ... | x |
    /// ...
    /// bin0 + ... + bin31 | x | x | x | x | x | x | x | x | x | ... | e |
    ///
    ///  where `e` stands for a valid accumulated minimum error value;
    ///  `x` for an unreachable sum because there is no combination of allowed steps
    ///  that can be summed up to match the target with the number of bins given.
    ///
    /// Since there are multiple combinations of summands, the one with the best total
    /// error is used as the cumulative error.
    ///
    /// bin0:
    ///  - these are just the allowed steps of 1, 2, 4, 8, 16, 32.
    ///
    /// bin0 + bin1:
    ///  - 1: not possible because both bins contribute at least 1.
    ///  - 2: 1 + 1
    ///  - 3: 1 + 2 or 2 + 1
    ///  - 4: 2 + 2
    ///  - 5: 1 + 4 or 4 + 1
    ///  - 6: 2 + 4 or 4 + 2
    ///  - 7: not possible, 1 + 6, 2 + 5, 4 + 3 all require an illegal number of points value.
    ///    ...
    ///
    /// Note: The shape of the table is of an upper triangle matrix of size 32 by 257.
    ///       The rows are also truncated on the right side through the maximum sum possible
    ///       in each row. There is significant room for space savings but at the cost of
    ///       a substantial increase in complexity which makes it not worth it for the modest
    ///       size of 32 * 257.
    ///
    /// Returns:
    /// - the ideal number of points as the power of 2 exponent (i.e. 5 -> 2^5 = 32).
    /// - the total cumulative error as specified by `weights` vs the fully populated grid.
    fn get_bin_npts(weights: &[[f32; 6]; 32]) -> ([u8; 32], f32) {
        // Dynamic programming approach; 32 bins, 256 points budget.
        // `f32::MAX` is the sentinel value indicating budget unreachable.
        let mut dp = [[(f32::MAX, 0_u8); 257]; 32];

        // Populate `dp` for the first bin, special case because we can't
        // check the one before.
        for (w_idx, err) in weights[0].iter().enumerate() {
            let npts = 1 << w_idx;
            dp[0][npts] = (*err, w_idx as u8);
        }

        // Populate `dp` for the other bins.
        for (bin_idx, weights) in weights.iter().enumerate().skip(1) {
            for (w_idx, err) in weights.iter().enumerate() {
                let npts = 1 << w_idx;

                // All previous bins contribute at least 1 to the sum;
                // and this one contributes `npts`.
                let min_budget = bin_idx + 1 + npts;
                // All previous bins contribute at most 32 points to the sum;
                // and this one `npts`; but the budget is at most 256.
                let max_budget = std::cmp::min((bin_idx + 1) * 32 + npts, 256);

                for budget in min_budget..=max_budget {
                    let new_err = match dp[bin_idx - 1][budget - npts] {
                        (f32::MAX, _) => continue,
                        (prev_err, _) => prev_err + err,
                    };
                    if let (this_err, _) = dp[bin_idx][budget]
                        && this_err > new_err
                    {
                        dp[bin_idx][budget] = (new_err, w_idx as u8);
                    }
                }
            }
        }

        // Backtracking through `dp` recovering the sequence of ideal `npts`.
        let mut bin_npts = [0; 32];
        let mut budget = 256;
        for bin_idx in (0..32).rev() {
            let (_, pow) = dp[bin_idx][budget];
            bin_npts[bin_idx] = pow;
            budget -= (1 << pow) as usize;
        }

        // Return the ideal number of points for each bin,
        // along with the corresponding total accumulated error.
        (bin_npts, dp[31][256].0)
    }

    /// Samples the ICDF stored in 32 bins with a non-uniform grid spacing in each bin
    /// (but 257 data points in total) using a `u16` that is mapped directly to [0, 1].
    ///
    /// The upper 5 bits (index 0 to 31) are used to find the correct bin; the lower
    /// `interp_bits` bits are used for linear interpolation, leaving exactly
    /// `11 - interp_bits` for indexing data points inside one bin.
    ///
    /// The process is completely branchless and based on bit manipulation and lookup
    /// tables.
    pub fn sample(&self, probe: u16) -> f32 {
        // This lookup table goes into the `.rodata` block of the
        // compiled binary and will sit in L1 cache while this function
        // is on the hot path.
        const NORMALIZERS: [f32; 6] = [
            1.0 / 63.0,   // 2^6 - 1
            1.0 / 127.0,  // 2^7 - 1
            1.0 / 255.0,  // 2^8 - 1
            1.0 / 511.0,  // 2^9 - 1
            1.0 / 1023.0, // 2^10 - 1
            1.0 / 2047.0, // 2^11 - 1
        ];

        // Use the highest 5 bits (2^5 = 32) as the index for `self.indexing`.
        // It's one out of 32 bins and then times 2 for 2 bytes in each bin,
        // hence the shift by 10 rather than 11; `bin_idx <= 62`.
        let bin_idx = (probe & 0xF800) >> 10;

        // Bounds checking can be safely avoided here because indexing happens
        // into an array of exactly 64 bytes with a 6 bit integer (2^6 = 64).
        let start_idx = unsafe { *self.indexing.get_unchecked(bin_idx as usize) };
        let interp_bits = unsafe { *self.indexing.get_unchecked(bin_idx as usize + 1) };
        debug_assert!(
            (6..=11).contains(&interp_bits),
            "6 <= interp_bits({interp_bits}) <= 11 violated"
        );

        // Index inside a specific bin; 0x07FF (this is NOT 0xF800) extracts
        // the lower 11 bits, these are shifted by the number of bits used for
        // interpolation, leaving 0 bits (offset 0) to 5 bits (offset up to 31).
        let offset = (probe & 0x07FF) >> interp_bits;
        // Total index in `self.vals`.
        let idx = start_idx + offset as u8;

        // Bounds checking can be safely avoided here because `idx` is a `u8`
        // and `self.vals` is an array of 257 elements.
        let f1 = unsafe { *self.vals.get_unchecked(idx as usize + 1) };
        let f0 = unsafe { *self.vals.get_unchecked(idx as usize) };

        // Extract the low interpolation bits; convert to `f32`; then normalize them
        // to generate a scaling constant in [0, 1];
        let mask = (1 << interp_bits) - 1;
        let interp = (probe & mask) as f32;
        let normalizer = unsafe { *NORMALIZERS.get_unchecked(interp_bits as usize - 6) };
        let s = interp * normalizer;

        // Fused instruction of `f0 + (f1 - f0) * s`.
        s.mul_add(f1 - f0, f0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    // Tolerance levels for basic float operations.
    const FTOL64: f64 = 1e-10;
    const FTOL32: f32 = 1e-5;
    // Tolerance level for integrations over a distribution.
    const INTEGRATION_TOL: f64 = 1e-2;

    fn check_linspace(xmin: f64, xmax: f64, count: usize) {
        let space = linspace(xmin, xmax, count);
        assert_eq!(count, space.len());
        assert_eq!(xmin, space[0]);
        assert_eq!(xmax, space[space.len() - 1]);
        // Check the diff is constant between all points, i.e. it's a linear spacing.
        let diff = space
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect::<Vec<_>>();
        assert!(diff.iter().all(|i| (*i - diff[0]).abs() < FTOL64));
    }

    #[test]
    fn test_linspace() {
        // Edge cases.
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(vec![0.0], linspace(0.0, 1.0, 1));

        // Something with integers.
        check_linspace(-1.0, 1.0, 3);
        // Something with ugly fractions.
        check_linspace(-2.0, 3.0, 13);
        // Something not aligned at all and also negative slope.
        check_linspace(2.1123, -0.2134, 345);
        // Something with no length at all, i.e. `xmin == xmax`.
        check_linspace(1.0, 1.0, 10);
    }

    #[test]
    #[should_panic]
    fn blow_up_linspace() {
        linspace(0.0, 1.0, 100_000_000);
    }

    /// Helper function for tests generating a normalized gaussian distribution,
    /// i.e. f(x) = 1/sqrt(pi) * exp(-x^2).
    ///
    /// Returns `(x, f, sigma_squared)`.
    fn normalized_gaussian(count: usize) -> (Vec<f64>, Vec<f64>) {
        let x = linspace(-3.0, 3.0, count);
        let f = x
            .iter()
            .map(|x| (-x * x).exp() / std::f64::consts::PI.sqrt())
            .collect::<Vec<_>>();
        // The value for the variance is less than 0.5 because of the limited domain.
        (x, f)
    }

    #[test]
    fn test_trapz() {
        assert_eq!(
            Err(DistributionError::LenMismatch),
            trapz(&[0.0, 1.0], &[1.0])
        );

        let (x, f) = normalized_gaussian(1000);
        let area = trapz(&x, &f).unwrap();
        assert!((area - 1.0).abs() < INTEGRATION_TOL);
    }

    #[test]
    fn test_cumtrapz() {
        assert_eq!(
            Err(DistributionError::LenMismatch),
            trapz(&[0.0, 1.0], &[1.0])
        );

        let (x, f) = normalized_gaussian(1000);
        let area = cumtrapz(&x, &f).unwrap();

        assert_eq!(1000, area.len());
        assert_eq!(0.0, area[0]);
        assert_matches!(area.last(), Some(a) if (a - 1.0).abs() < INTEGRATION_TOL);
    }

    #[test]
    fn test_cdf_from_distribution() {
        // Check uneven length error.
        assert_eq!(
            Err(DistributionError::LenMismatch),
            cdf_from_distribution(&[0.0, 1.0, 2.0], &[0.0, 1.0]),
        );

        // Check underpopulated error for len 1 and 2.
        assert_eq!(
            Err(DistributionError::Underpopulated),
            cdf_from_distribution(&[], &[]),
        );
        assert_eq!(
            Err(DistributionError::Underpopulated),
            cdf_from_distribution(&[1.0], &[1.0]),
        );
        assert!(cdf_from_distribution(&[0.0, 1.0], &[0.0, 1.0]).is_ok());

        // Check a bad distribution with negative values.
        let x = linspace(-2.0, 2.0, 10_000);
        let f = x.iter().map(|x| x * (-x * x).exp()).collect::<Vec<_>>();
        assert_eq!(
            Err(DistributionError::NotMonotonic),
            cdf_from_distribution(&x, &f),
        );

        // Check a normalized test gaussian (the good case).
        let (x, f) = normalized_gaussian(10_000);
        let cdf = cdf_from_distribution(&x, &f).unwrap();
        assert_eq!(10_000, cdf.len());
        // `cdf` should be perfectly normalized, starting at exactly 0.0...
        assert_eq!(Some(&0.0), cdf.first());
        // ...and ending at exaclty 1.0.
        assert_eq!(Some(&1.0), cdf.last());
    }

    #[test]
    fn test_get_inv_cdf() {
        let (x, f) = normalized_gaussian(10_000);
        let cdf = cdf_from_distribution(&x, &f).unwrap();
        let cdf_inv = ICDFSampler16::get_cdf_inv(&x, &cdf);

        // Check it starts with exactly `x.first()`, ...
        assert_eq!(x[0] as f32, cdf_inv[0]);
        // ...ends with exactly `x.last()` and ...
        assert_eq!(*x.last().unwrap() as f32, cdf_inv[1024]);
        // ... is basically 0.0 in the center.
        assert!(cdf_inv[512].abs() < FTOL32);

        // Check F^-1(u) (`cdf_inv`) is strictly monotonically increasing.
        assert!(
            cdf_inv
                .iter()
                .zip(cdf_inv.iter().skip(1))
                .all(|(f0, f1)| f0 < f1)
        );

        // For a gaussian, F^-1(u) (`cdf_inv`) should be antisymmetric around
        // u = 1/2.
        assert!(
            cdf_inv[..512]
                .iter()
                .zip(cdf_inv[512..].iter().rev())
                .all(|(fplus, fminus)| (fplus + fminus).abs() < FTOL32)
        );
    }

    #[test]
    fn test_get_error_weights() {
        let (x, f) = normalized_gaussian(10_000);
        let cdf = cdf_from_distribution(&x, &f).unwrap();
        let cdf_inv = ICDFSampler16::get_cdf_inv(&x, &cdf);
        let weights =
            ICDFSampler16::get_error_weights(&cdf_inv, |f_interp, f_ref| (f_interp - f_ref).abs());

        // The weights for 32 points in each bin is exactly 0.0.
        assert!(weights.iter().all(|w| w[5] == 0.0));

        // For a normalized gaussian, the computed weights should be symmetric, i.e.
        // the first 16 bins are mirrors of the last 16 bins.
        for pts_idx in 0..5 {
            assert!(
                weights[..16]
                    .iter()
                    .zip(weights[16..].iter().rev())
                    .all(|(wm, wp)| (wm[pts_idx] - wp[pts_idx]).abs() < FTOL32)
            );
        }

        // For a normalized gaussian, the errors in each bin should be strictly
        // monotonically decreasing with increasing points.
        // Note: The general case is only monotonically decreasing because extended
        //       linear segments do not lose precision when thinning out points.
        for weights in weights.iter() {
            assert!(
                weights
                    .iter()
                    .zip(weights.iter().skip(1))
                    .all(|(this_weight, next_weight)| *this_weight > *next_weight)
            );
        }
    }

    #[test]
    fn test_get_bin_npts() {
        let (x, f) = normalized_gaussian(10_000);
        let cdf = cdf_from_distribution(&x, &f).unwrap();
        let cdf_inv = ICDFSampler16::get_cdf_inv(&x, &cdf);
        let weights =
            ICDFSampler16::get_error_weights(&cdf_inv, |f_interp, f_ref| (f_interp - f_ref).abs());
        let (bin_npts, total_err) = ICDFSampler16::get_bin_npts(&weights);

        // This error is known for these settings and not derived from first principles;
        // it is an empirical check.
        assert!(total_err < 0.015);
        // For a normalized gaussian the number of points per bin should be symmetric
        // in the first half vs the second half.
        assert!(
            bin_npts
                .iter()
                .take(16)
                .zip(bin_npts.iter().rev().take(16))
                .all(|(n0, n1)| n0 == n1)
        );

        // The total sum of points across all bins must be 256.
        assert_eq!(
            256_usize,
            bin_npts.iter().map(|pow| (1 << *pow) as usize).sum()
        );
    }

    #[test]
    fn test_sampler_with_direct_error() {
        let (x, f) = normalized_gaussian(10_000);
        let cdf = cdf_from_distribution(&x, &f).unwrap();
        let (sampler, total_err) = ICDFSampler16::from_symmetric_mean_relative_errors(&x, &cdf);

        // We will insist the total error is not more than 5% for all the values
        // vs the input cdf resampled. Note: there is always already some error
        // in the input and this resampling, but on the order of a couple % at most.
        assert!(total_err < 0.05);

        // Indices must grow in correspondence with the number of points in
        // each bin, i.e. the sum of `2^pow`...
        let mut jdx = 0;
        for w in sampler.indexing.windows(2).step_by(2) {
            match w {
                [idx, interp_bits] => {
                    assert_eq!(jdx as u8, *idx);
                    let indexing_bits = 11 - interp_bits;
                    jdx += 1 << indexing_bits;
                }
                _ => unreachable!(),
            }
        }
        // ...and the total sum must be 256.
        assert_eq!(256, jdx);

        // The data points for a gaussian should be anti symmetric up to some tolerance...
        assert!(
            sampler.vals[..128]
                .iter()
                .zip(sampler.vals[128..].iter().rev())
                .all(|(fm, fp)| (fm + fp).abs() < FTOL32)
        );
        // And since 257 is odd, it should include 0 in the center.
        assert!(sampler.vals[128].abs() < FTOL32);

        // Check the edges are included.
        assert_eq!(x[0] as f32, sampler.sample(0));
        assert_eq!(x[9_999] as f32, sampler.sample(u16::MAX));

        // Compute the empirical variance and compare it to the theorical value.
        let emp_variance = 1.0 / ((u16::MAX - 1) as f64)
            * (0..=u16::MAX)
                .into_iter()
                .map(|p| {
                    let s = sampler.sample(p) as f64;
                    s * s
                })
                .sum::<f64>();
        // Some deviation from the theoretical value (0.5) is expected but below 1%.
        assert!((emp_variance / 0.5 - 1.0).abs() < 1e-2,);
    }
}
