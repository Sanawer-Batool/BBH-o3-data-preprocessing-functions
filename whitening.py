import numpy as np
from load_data import load_o3_strain
from psd import compute_psd
from numpy.typing import NDArray

def whiten(
        signal: NDArray[np.float64],
        psd: NDArray[np.float64],
        freqs:NDArray[np.float64],
        sample_rate: int,
        highpass_hz: float=30.0,
        lowpass_hz: float=500.0
):
    # removing DC offset
    signal = signal - np.mean(signal)

    # FFT of the signal
    H = np.fft.rfft(signal)

    # interpolate the PSD to match the frequencies of the FFT
    signal_freqs = np.fft.rfftfreq(len(signal), d = 1.0/sample_rate)
    psd_interpolated = np.interp(signal_freqs, freqs, psd)

    # highpass and low pass filters
    psd_interpolated[signal_freqs < highpass_hz] = 1e10
    psd_interpolated[signal_freqs > lowpass_hz] = 1e10

    # Compute the ASD and divide
    asd = np.sqrt(psd_interpolated + 1e-10)
    H_whitened = H / asd

    # Go back to time domain with inverse FFT
    whitened = np.fft.irfft(H_whitened, n = len(signal))

    # Crop the edges
    pad = int(0.5 * sample_rate)
    whitened = whitened[pad : -pad]

    print(f"Length of signal before cropping: {len(signal)}")
    print(f"Length of signal after cropping: {len(whitened)}")
    print(f"Mean of whitened signal: {np.mean(whitened)}")
    print(f"Std of whitened signal: {np.std(whitened)}")

    return(
        {
            "whitened_signal": whitened,
            "sample_rate": sample_rate,
            "highpass_hz": highpass_hz,
            "lowpass_hz": lowpass_hz,
            "n_cropped": 2 * pad # how many samples were cropped
        }
    )
