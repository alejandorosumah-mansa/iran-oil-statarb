"""Brownian bridge interpolation for static segments in basket levels.

The S3 basket level is recorded at monthly frequency but stored on a
daily grid, creating long runs of identical values between rebalance
dates. A Brownian bridge injects stochastic variation consistent with
observed endpoint-to-endpoint moves, producing a plausible daily path.

Bridge formula
--------------
For a segment [t0, t1] with start value a and end value b:

    B(t) = a + (t - t0)/(t1 - t0) * (b - a) + sigma * W_bridge(t)

where W_bridge(t) is a standard Brownian bridge (pinned to 0 at both
endpoints) and sigma is estimated from the absolute change |b - a|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SegmentInfo:
    """Metadata for a single static (flat) segment."""

    start_idx: int
    end_idx: int
    start_val: float
    end_val: float
    length: int


def detect_static_segments(
    series: pd.Series,
    min_run: int = 5,
    tol: float = 1e-6,
) -> List[SegmentInfo]:
    """Find contiguous runs where the series does not change.

    Parameters
    ----------
    series : pd.Series
        Numeric series to scan.
    min_run : int
        Minimum number of consecutive identical values to qualify
        as a static segment.
    tol : float
        Absolute tolerance for comparing successive values.

    Returns
    -------
    list[SegmentInfo]
        One entry per detected segment.
    """
    values = series.values.astype(float)
    n = len(values)
    segments: List[SegmentInfo] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(values[j] - values[i]) < tol:
            j += 1
        run_length = j - i
        if run_length >= min_run:
            # Determine endpoint value: use next distinct value if available
            end_val = values[j] if j < n else values[j - 1]
            segments.append(
                SegmentInfo(
                    start_idx=i,
                    end_idx=j - 1,
                    start_val=values[i],
                    end_val=end_val,
                    length=run_length,
                )
            )
            i = j
        else:
            i += 1
    return segments


def brownian_bridge(
    n: int,
    start_val: float,
    end_val: float,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a discrete Brownian bridge path of length n.

    The path is pinned to start_val at t=0 and end_val at t=n-1.

    Parameters
    ----------
    n : int
        Number of points (including endpoints).
    start_val : float
        Value at t=0.
    end_val : float
        Value at t=n-1.
    sigma : float
        Volatility scaling for the bridge noise.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of length n with the bridge path.
    """
    if n < 2:
        return np.array([start_val])
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0.0, 1.0, n)
    # Linear interpolation from start to end
    linear = start_val + t * (end_val - start_val)

    # Standard Brownian bridge noise (pinned to 0 at both endpoints)
    dt = 1.0 / (n - 1)
    increments = rng.normal(0, np.sqrt(dt), size=n - 1)
    w = np.zeros(n)
    w[1:] = np.cumsum(increments)
    # Pin to zero at t=1
    w -= t * w[-1]

    return linear + sigma * w


def apply_brownian_bridge(
    series: pd.Series,
    min_run: int = 5,
    tol: float = 1e-6,
    sigma_scale: float = 0.3,
    seed: int | None = None,
) -> Tuple[pd.Series, List[SegmentInfo]]:
    """Replace all static segments in a series with Brownian bridge paths.

    Parameters
    ----------
    series : pd.Series
        Original series (e.g. S3 basket level).
    min_run : int
        Minimum flat-run length to interpolate.
    tol : float
        Tolerance for detecting flat runs.
    sigma_scale : float
        Fraction of |end - start| used as bridge sigma. When the
        segment is truly flat (start == end), a small default sigma
        proportional to the level is used instead.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    tuple[pd.Series, list[SegmentInfo]]
        - Bridged series (same index as input).
        - List of segment metadata.
    """
    rng = np.random.default_rng(seed)
    segments = detect_static_segments(series, min_run=min_run, tol=tol)
    bridged = series.copy().astype(float)

    for seg in segments:
        delta = abs(seg.end_val - seg.start_val)
        if delta < tol:
            # Flat segment with no level change - use small noise
            sigma = seg.start_val * 0.001
        else:
            sigma = delta * sigma_scale
        path = brownian_bridge(
            n=seg.length,
            start_val=seg.start_val,
            end_val=seg.end_val,
            sigma=sigma,
            rng=rng,
        )
        bridged.iloc[seg.start_idx : seg.end_idx + 1] = path

    return bridged, segments
