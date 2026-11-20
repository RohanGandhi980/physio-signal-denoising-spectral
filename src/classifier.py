import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from data_laoder import load_ecg_data, load_eeg_subjects
from filters import design_iir_bandpass, design_notch
from spectral import compute_relative_band_power

def plot_psd(epochs_open, epochs_closed, fs):
    from scipy.signal import welch
    f, psd_open = welch(epochs_open, fs=fs, nperseg=int(fs*2))
    f, psd_closed = welch(epochs_closed, fs=fs, nperseg=int(fs*2))

    plt.figure(figsize=(8,5))
    plt.plot(f, np.mean(psd_open, axis=(0,1)), label='Eyes Open (Low Alpha)')
    plt.plot(f, np.mean(psd_closed, axis=(0,1)), label='Eyes Closed (High Alpha)')

    plt.xlim(1,45)
    plt.axvspan(8,12,color='gray', alpha=0.2, label='Alpha Band (8-12Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Occipital EEG: Alpha Blocking Phenomenon')
    plt.legend()
    plt.grid(True)
    plt.savefig('alpha_blocking_psd.png')
    print('Figure saved at "alpha_blocking_psd"')


def main():
    print('Physiological EEG Siganl Denoising')
    raw_list, labels_list = load_eeg_subjects(subjects=range(1,11))

    X_list, y_list = [], []
    target_channels = ["O1..", "Oz..", "O2.."] # Occipital channels (visual cortex)
    
    epochs_open_list = []
    epochs_closed_list = []

    print('\n Processing DSP Pipeline for all Subjects')
    for raw, label in zip(raw_list, labels_list):
        raw.pick(target_channels)
        fs = raw.info['sfreq']
        data = raw.get_data()

        #filter design 
        b_notch, a_notch = design_notch(fs, f_notch=50.0)
        a_iir, b_iir = design_iir_bandpass(fs, 1.0, 45.0)

        #apply filters
        filtered = signal.filtfilt(b_notch, a_notch, data, axis=-1)
        filtered = signal.filtfilt(a_iir, b_iir, filtered, axis=-1)

        #epochs segment 
        epoch_len = int(fs*2)
        n_epochs = filtered.shape[-1] // epoch_len
        epochs = filtered[:, : n_epochs * epoch_len].reshape(len(target_channels), n_epochs, epoch_len)
        epochs = np.transpose(epochs, (1, 0, 2)) # shape: (n_epochs, n_channels, n_samples)

        #specteral feature extraction 
        feats = compute_relative_band_power(epochs, fs, window='hamming')
        X_list.append(feats)
        y_list.append(np.full(n_epochs, label))
        if label == 0:
            epochs_open_list.append(epochs)
        else:
            epochs_closed_list.append(epochs)

    #machine learning
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print("\nTraining SVM Classifier (5 Fold Cross Validation)")
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')

    print(f"Results across 10 subjects")
    print(f"Accuracy: {acc_scores.mean()} ((± {acc_scores.std()})")
    print(f"ROC-AUC: {auc_scores.mean()} ((± {auc_scores.std()})")

    #generating visualizations 
    all_open = np.vstack(epochs_open_list)
    all_closed = np.vstack(epochs_closed_list)
    plot_psd(all_open, all_closed, fs)

if __name__ == "__main__":
    main()




