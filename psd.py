#PSD: How densely packed is the power around each frequency?
import numpy as np
from numpy.typing import NDArray
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gwosc import datasets

# GOAL: Amplitude → Energy → Power → PSD

def compute_psd(signal:NDArray[np.float64],
                sample_rate: int,
                fft_duration: float=2.0,
                overlap_dur:float=1.0):
    #covert time based input to sample counts
    nperseg = int(fft_duration * sample_rate) #number of samples in each FFT segment. With a sample rate of 2048 Hz and an FFT duration of 2 seconds, we have 4096 samples per segment.
    nstride = nperseg - int(overlap_dur * sample_rate) #the stride determines how much we shift the window for the next FFT. With 50% overlap, we shift by half the window size.

    if len(signal) < nperseg:
        raise ValueError(
            f"Signal too short: {len(signal)} samples, need at least {nperseg} for one window."
    )

    # Hann window 
    window = np.hanning(nperseg)

    power_array = []

    # overlapping windows + FFT
    for i in range(0, len(signal) - nperseg + 1, nstride): # Only allow starting positions where a FULL window still fits
        segment = signal[i : i + nperseg] # Take a chunk of time-domain data. You still have:,amplitude vs time
        windowed = segment * window # Apply the Hann window to the segment. This helps to reduce spectral leakage in the FFT.
        fft_vals = np.fft.rfft(windowed) # Now you know what frequencies exist.
        power = np.abs(fft_vals)**2 # OR fft_vals.real**2 + fft_vals.imag**2 -> How much of each frequency is present in this signal chunk?
        power_array.append(power)

    psd_raw = np.mean(power_array, axis = 0)

    # Normalize by the sampling frequency and the window power to get the PSD in units of power per Hz
    scale = 1.0/(sample_rate * np.sum(window**2))
    psd = psd_raw * scale

    if not np.all(psd >= 0):
        raise ValueError("PSD contains negative values — something went wrong in power computation.")


    # Compute the frequency axis
    freqs = np.fft.rfftfreq(nperseg, d=1.0/sample_rate)

    k = len(power_array)
    print(f"Number of windows computed: {k}")

    freq_resolution = sample_rate / nperseg
    print(f"Frequency Resolution: {freq_resolution} Hz per bin")

    idx_100hz = int(100 / freq_resolution)
    print(f"PSD at 100 Hz: {psd[idx_100hz]:.6e}")

    print(f"Total frequency bins: {len(freqs)}")

    return(
        {
            "psd": psd,
            "freqs": freqs,
            "sample_rate": sample_rate,
            "nperseg": nperseg,
            "window": window
        }
    )







