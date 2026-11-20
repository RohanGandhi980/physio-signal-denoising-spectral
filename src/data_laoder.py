import mne
import wfdb 
import numpy as np 
import os 

def load_ecg_data(record_name='100', sampfrom=0, sampto='3000'):
    print(f"Looking for EEG Data from {record_name}")
    record = wfdb.rdrecord(record_name, pn_dir='mitdb', sampfrom=sampfrom, sampto=sampto)
    #extracting channel 0
    eeg_signal = record.p_signal[:,0]
    fs = record.fs
    return eeg_signal, fs

def load_eeg_subjects(subjects=(1,), runs=(1,2)):
    mne.set_log_level('WARNING')
    raw_list = []
    labels = []
    for subj in subjects: 
        for run in runs:
            print(f"Looking for EEG Subjects {subj}, Runs {run}")
            #downloads the data to /mne-data
            file_paths = mne.datasets.eegbci.load_data(subj, runs=[run])
            raw = mne.io.read_raw_edf(file_paths[0], preload=True)

            #label 0 for Run 1 (eye open), and label 1 for Run 2 (eyes closed)
            label = 0 if run == 1 else 1
            raw_list.append(raw)
            labels.append(label)

        return raw_list, labels
    
if __name__ == "__main__":
    #test the loaders here
    ecg, fs_ecg = load_ecg_data(sampto=1000)
    print(f"Loaded ECG: {len(ecg)} samples at {fs_ecg} Hz")




