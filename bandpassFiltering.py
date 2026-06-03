import numpy as np
from load_data import load_o3_strain
from psd import compute_psd
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

def bandpass_filter(
        signal: NDArray[np.float64],
        sample_rate: int,
        lowcut_hz: float=30.0,
        highcut_hz: float=500.0,
        order: int=4
):
    nyquist = sample_rate / 2
    low_norm = lowcut_hz / nyquist
    high_norm = highcut_hz / nyquist
    if not (0 < low_norm < 1) or not (0 < high_norm < 1):
        raise ValueError(f"Normalised frequencies must be strictly between 0 and 1. Got low={low_norm:.4f}, high={high_norm:.4f}")
    
    if low_norm >= high_norm:
        raise ValueError(
            f"lowcut_hz ({lowcut_hz}) must be less than highcut_hz ({highcut_hz})."
    )

    b, a = butter(order, [low_norm, high_norm], btype="band") # btype='band' means bandpass (keep frequencies between low and high)

    filtered = filtfilt(b, a, signal)
    if len(signal) != len(filtered):
        raise ValueError("Length of data does not match.")
    if np.std(filtered) > np.std(signal):
        raise ValueError("Std of the filtered signal should be less than or equal to the std of the input signal.")
    if not np.all(np.isfinite(filtered)):
        raise ValueError("Filtered signal contains NaN or Inf values.")
    
    print(f"The cut off frequency used: {lowcut_hz} Hz to {highcut_hz} Hz")
    print(f"low_norm filter is: {low_norm}, and high_norm filter is: {high_norm}")
    print(f"Std before filter: {np.std(signal)}, Std after filter: {np.std(filtered)}")
    if len(signal) == len(filtered):
        print("Length of signal is preserved")
    
    return(
        {
            "filtered_signal": filtered,
            "sample_rate": sample_rate,
            "lowcut_hz":lowcut_hz,
            "highcut_hz": highcut_hz,
            "filter_order": order
        }
    )
    