import numpy as np
from scipy import signal, integrate

def compute_relative_band_power(epochs, fs, window='hann'):
    """
    Extracts relative Alpha, Beta, and Gamma power using Welch's method.
    epochs shape: (n_epochs, n_channels, n_samples)
    """
    bands = {
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0)
    }
    
    # 2-second segments -> 0.5 Hz frequency resolution
    nperseg = int(fs * 2.0)
    if nperseg > epochs.shape[-1]:
        nperseg = epochs.shape[-1]
        
    freqs, psd = signal.welch(epochs, fs=fs, window=window, nperseg=nperseg, axis=-1)
    
    # Total power between 1 and 45 Hz
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = integrate.simpson(psd[..., total_mask], x=freqs[total_mask], axis=-1)
    
    features = []
    # Safely unpack the dictionary items
    for band_name, limits in bands.items():
        fmin = limits[0]
        fmax = limits[1]
        
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = integrate.simpson(psd[..., mask], x=freqs[mask], axis=-1)
        rel_power = band_power / (total_power + 1e-12)
        features.append(rel_power)
        
    # Concatenate band powers -> shape: (n_epochs, n_channels * 3)
    return np.concatenate(features, axis=-1)