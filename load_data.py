import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from gwosc.datasets import event_gps
from gwosc import datasets


# strain_data = datasets.find_datasets()
# print(strain_data)

def load_o3_strain(event_name:str, detector:str, duration_sec:float):
    if(duration_sec <= 0):
        raise ValueError("Duration must be a positive number.")
    gps_center = event_gps(event=event_name)

    start_gps = gps_center - (duration_sec/2)
    end_gps = gps_center + (duration_sec/2)

    if (detector not in ['H1', 'L1', 'V1']):
        raise ValueError("Invalid detector. Please choose from 'H1', 'L1', 'V1'.")
    
    strain_data = TimeSeries.fetch_open_data(detector, start=start_gps, end=end_gps, cache=True)

    strain_data = strain_data.resample(2048)

    sample_rate = strain_data.sample_rate.value

    expected_samples = int(duration_sec * sample_rate)

    if abs(len(strain_data) - expected_samples) > 2:
        raise ValueError(f"Expected {expected_samples} samples, but got {len(strain_data)} samples. Check the duration and sample rate.")
    
    raw_signal = strain_data.value

    if not np.all(np.isfinite(raw_signal)):
        raise ValueError(
            "Downloaded strain data contains NaN or infinite values."
        )
    
    rms = float(np.sqrt(np.mean(raw_signal**2)))
    if rms>0:
        scaled_signal= raw_signal/rms
    else:
        raise ValueError("RMS of downloaded strain data is zero, cannot scale the signal.")
    
    print(f"Loaded strain data for event {event_name} from detector {detector} with duration {duration_sec} seconds.")
    print(f"Sample rate of the loaded signal: {strain_data.sample_rate}")
    print(f"GPS center of the loaded signal: {gps_center}")
    print(f"Scale factor used for normalization: {rms}")
    print(f"Expected number of samples: {expected_samples}")

    return(
        {
            "signal": scaled_signal,
            "duration_sec": duration_sec,
            "sample_rate":sample_rate,
            "expected_samples": expected_samples,
            "event_name":event_name,
            "detector":detector,
            "gps_center":gps_center,
            "scale_factor": rms
        }
    )

load_o3_strain(event_name="GW190521", detector="H1", duration_sec=4)

