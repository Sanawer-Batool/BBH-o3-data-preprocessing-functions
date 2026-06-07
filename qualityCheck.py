import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis

def quality_check(
        signal: NDArray[np.float64],
        sample_rate: int,
        snr_threshold: float=3.0,
        flatness_threshold: float=0.01,
        max_allowed_std: float=5.0
):
    # 1
    has_nan_inf = not np.all(np.isfinite(signal))
    if has_nan_inf:
        print("Warning! FAILED signal, it contains NAN or Infinite values.")
    
    # 2
    flatness_ratio = np.std(signal) / 1.0
    is_flat = flatness_ratio < flatness_threshold
    if is_flat:
        print("warning! The signal is essentially flat.")
    
    # 3
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    snr_proxy = (peak / rms) if rms > 0 else 0.0
    
    if snr_proxy >= snr_threshold:
        print("Worth investigating the signal, as proxy_snr is >= 3")
    
    # 4
    max_abs = np.max(np.abs(signal))
    is_glitch_candidate = max_abs > max_allowed_std
    if is_glitch_candidate:
        print("warning! this event seems like a glitch candidate")

    # 5
    kurt = kurtosis(signal, fisher=False)
    high_kurt = kurt > 10
    if high_kurt:
        print("warning! heavy non-Guassian tails detected, could be glith")
    
    report = {
        "has_nan_inf": has_nan_inf,
        "is_flat": flatness_ratio < flatness_threshold,
        "snr_proxy": snr_proxy,
        "is_interesting": snr_proxy >= snr_threshold,
        "is_glitch_candidate": is_glitch_candidate,
        "max_abs_value": float(max_abs),
        "kurtosis": float(kurt),
        "high_kurtosis": kurt > 10
    }

    # overall
    if report["has_nan_inf"] or report["is_flat"]:
        verdict = "REJECT"

    elif report["is_glitch_candidate"] or report["high_kurtosis"]:
        verdict = "SUSPECT"

    elif report["is_interesting"]:
        verdict = "INTERESTING"

    else:
        verdict = "CLEAN"

    report["verdict"] = verdict

    return report
