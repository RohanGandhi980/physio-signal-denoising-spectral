

# Physiological Signal Denoising & Spectral Analysis (EEG)

This repository contains an end-to-end Digital Signal Processing (DSP) and Machine Learning pipeline designed to extract, filter, and classify raw neurophysiological data.

Built using standard scientific computing libraries (`scipy`, `mne`, `scikit-learn`), this project demonstrates the isolation of neural frequency bands from noisy time-series data and proves the physiological "Alpha Blocking" phenomenon via Support Vector Machine (SVM) classification.

## 🧠 Project Architecture

The pipeline is divided into four modular stages:

1. **Automated Data Streaming (`data_loader.py`):**
Programmatically fetches the open-access **PhysioNet EEG Motor Movement/Imagery Dataset**. Extracts Run 1 (Eyes Open) and Run 2 (Eyes Closed) resting-state data across 10 subjects, focusing strictly on occipital channels (O1, Oz, O2) situated over the visual cortex.
2. **Digital Signal Processing & Denoising (`filters.py`):**
Raw neural time-series data is heavily corrupted by artifacts. The signal is cleaned using zero-phase forward-backward filtering (`filtfilt`) to prevent the temporal phase distortion common in causal IIR filters:
* **50 Hz IIR Notch Filter:** Eradicates European powerline interference.
* **1–45 Hz Butterworth Bandpass Filter:** Removes slow baseline wander (sweat/motion artifacts) and high-frequency electromyographic (muscle) noise.


3. **Spectral Feature Extraction (`spectral.py`):**
Transitions the cleaned time-domain data into the frequency domain using **Welch’s Power Spectral Density (PSD)**. Evaluates 2-second overlapping epochs (Hamming window) to extract relative power in the Alpha (8–12 Hz), Beta (13–30 Hz), and Gamma (30–45 Hz) bands.
4. **Machine Learning Classification (`classifier.py`):**
A Support Vector Machine (RBF Kernel) is trained on the extracted spectral features to automatically classify the subject's state (Eyes Open vs. Eyes Closed) based solely on brainwave morphology.

## 📊 Results & Performance

By analyzing the spectral leakage and relative band power, the pipeline easily isolates the **Alpha Blocking Phenomenon**—a well-documented neurophysiological event where occipital alpha rhythms surge when visual input is cut off (eyes closed) and desynchronize (block) when the eyes open.

### Classification Metrics (10 Subjects, 5-Fold Cross-Validation)

* **Accuracy:** 96.6% ($\pm$ 4.08%)
* **ROC-AUC:** 1.00 ($\pm$ 0.00)

### Power Spectral Density (PSD)


*Figure 1: Average Welch PSD across all occipital epochs. The graph proves a distinct, high-amplitude peak strictly within the 8–12 Hz Alpha band during the Eyes Closed state, contrasting with profound alpha attenuation during the Eyes Open state.*

## 🚀 How to Run

1. Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/physio-signal-denoising-spectral.git
cd physio-signal-denoising-spectral
pip install -r requirements.txt

```


2. Execute the main pipeline:
```bash
python src/classifier.py

```


*Note: The `mne` library will automatically download and cache the necessary PhysioNet EDF files on the first run.*

## 🛠 Dependencies

* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `mne`
* `wfdb`