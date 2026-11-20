import numpy as np 
from scipy import signal

def inject_noise(clean_signal, fs, snr_db=5.0):

    t = np.arange(len(clean_signal))/fs

    #50Hz powerline inference 
    powerline = 0.5 * np.sin(2*np.pi*50*t)
    #0.3 Hz respitory baseline wander
    baseline = 0.8 * np.sin(2*np.pi*0.3*t)

    #guassian white noise
    signal_power = np.mean(clean_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    white_noise = np.random.normal(0, np.sqrt(noise_power), len(clean_signal))

    return clean_signal + powerline + baseline + white_noise

def design_iir_bandpass(fs, f_low, f_high, order=4):
    #causal iir butterworth filter which is prone to non linear distortion
    nyq = 0.5*fs 
    b, a = signal.butter(order, [f_low/nyq, f_high/nyq], btype='bandpass')
    return b, a

def design_fir_bandpass(fs, f_low, f_high, numtaps=101, window='hamming'):
    #fir butterworth filter (in linear phase) which preserves waveform morphology
    b = signal.firwin(numtaps, [f_low, f_high], pass_zero=False, window=window, fs=fs)
    a = [1,0]
    return b,a 

def design_notch(fs, f_notch=50.0,Q=30.0):
    #iir notch to remove powerline baseline interference
    b, a = signal.iirnotch(f_notch, Q, fs)
    return b,a 

def compute_snr(clean, test):
    #calculates snr improvement in db
    noise = test - clean 
    return 10.0* np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-12))


