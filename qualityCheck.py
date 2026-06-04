import numpy as np
from numpy.typing import NDArray

def quality_check(
        signal: NDArray[np.float64],
        sample_rate: int,
        snr_threshold: float=3.0,
        flatness_threshold: float=0.01,
        max_allowed_std: float=5.0
):
    if not np.all(np.isfinite(signal)):
        raise ValueError("FAILED signal, it contains NAN or Infinite values.")
    
    flatness_ratio = np.std(signal) / 1.0
    if flatness_ratio < flatness_threshold:
        raise ValueError("The signal is essentially flat.")
    
    