import numpy as np
from numpy.typing import NDArray

def zscore_normalise(
        signal: NDArray[np.float64],
        epsilon: float = 1e-8
):
    mu = np.mean(signal)

    sigma = np.std(signal)

    sigma_safe = sigma + epsilon

    normalized = (signal - mu) / sigma_safe

    if abs(np.mean(normalized)) > 1e-6:
        raise ValueError(f"Normalisation failed: mean={np.mean(normalized):.2e}, expected ≈ 0")

    if abs(np.std(normalized) - 1.0) > 0.01:
        raise ValueError(f"Normalisation failed: std={np.std(normalized):.4f}, expected ≈ 1")

    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalised signal contains NaN or Inf values.")
    
    print(f"Mean before normalization: {mu}")
    print(f"Std deviation before normalization: {sigma_safe}")
    print(f"Mean after normalization: {np.mean(normalized)}")
    print(f"Std deviation after normalization: {np.std(normalized)}")
    if sigma < 1e-4:
        print(f"Warning: near-zero std detected, epsilon is dominant, signal may be flat.")
    
    return(
        {
            "normalized_signal": normalized,
            "mean_removed": mu,
            "std_used": sigma_safe,
            "epsilon": epsilon
        }
    )