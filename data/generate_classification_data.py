import numpy as np
from data_input import get_rms, random_signal
from sklearn.model_selection import train_test_split

def generate_mixed_signals(EEG_all, EOG_all, EMG_all, samples_per_class):
    # Ensure we have enough unique samples
    assert EEG_all.shape[0] >= samples_per_class * 5, "Not enough EEG samples"
    assert EOG_all.shape[0] >= samples_per_class * 5, "Not enough EOG samples"
    assert EMG_all.shape[0] >= samples_per_class * 5, "Not enough EMG samples"
    
    # Split the original data into 5 non-overlapping parts
    eeg_splits = np.array_split(EEG_all, 5)
    eog_splits = np.array_split(EOG_all, 5)
    emg_splits = np.array_split(EMG_all, 5)
    
    # Generate clean EEG (class 0)
    clean_eeg = eeg_splits[0][:samples_per_class]
    
    # Generate EEG + EOG (class 1)
    eog_eeg = []
    for i in range(samples_per_class):
        eeg = eeg_splits[1][i]
        eog = eog_splits[0][i]
        snr = np.random.uniform(-7, 2)
        coe = get_rms(eeg) / (get_rms(eog) * (10 ** (0.1 * snr)))
        mixed = eeg + eog * coe
        eog_eeg.append(mixed)
    eog_eeg = np.array(eog_eeg)
    
    # Generate EEG + EMG (class 2)
    emg_eeg = []
    for i in range(samples_per_class):
        eeg = eeg_splits[2][i]
        emg = emg_splits[0][i]
        snr = np.random.uniform(-7, 2)
        coe = get_rms(eeg) / (get_rms(emg) * (10 ** (0.1 * snr)))
        mixed = eeg + emg * coe
        emg_eeg.append(mixed)
    emg_eeg = np.array(emg_eeg)
    
    # Generate EEG + EMG + small EOG (class 2)
    emg_eog_eeg = []
    for i in range(samples_per_class):
        eeg = eeg_splits[3][i]
        emg = emg_splits[1][i]
        eog = eog_splits[1][i]
        snr_emg = np.random.uniform(-7, 2)
        snr_eog = np.random.uniform(-2, 2)  # Smaller EOG noise
        coe_emg = get_rms(eeg) / (get_rms(emg) * (10 ** (0.1 * snr_emg)))
        coe_eog = get_rms(eeg) / (get_rms(eog) * (10 ** (0.1 * snr_eog)))
        mixed = eeg + emg * coe_emg + eog * coe_eog
        emg_eog_eeg.append(mixed)
    emg_eog_eeg = np.array(emg_eog_eeg)
    
    # Generate EEG + small EMG + EOG (class 1)
    small_emg_eog_eeg = []
    for i in range(samples_per_class):
        eeg = eeg_splits[4][i]
        emg = emg_splits[2][i]
        eog = eog_splits[2][i]
        snr_emg = np.random.uniform(-2, 2)  # Smaller EMG noise
        snr_eog = np.random.uniform(-7, 2)
        coe_emg = get_rms(eeg) / (get_rms(emg) * (10 ** (0.1 * snr_emg)))
        coe_eog = get_rms(eeg) / (get_rms(eog) * (10 ** (0.1 * snr_eog)))
        mixed = eeg + emg * coe_emg + eog * coe_eog
        small_emg_eog_eeg.append(mixed)
    small_emg_eog_eeg = np.array(small_emg_eog_eeg)
    
    # Combine all signals with equal number of samples per class
    signals = []
    labels = []
    
    # Class 0: Clean EEG
    signals.append(clean_eeg)
    labels.extend([0] * samples_per_class)
    
    # Class 1: EOG contaminated (combining pure EOG and EOG dominant)
    signals.append(eog_eeg)
    signals.append(small_emg_eog_eeg)
    labels.extend([1] * (2 * samples_per_class))
    
    # Class 2: EMG contaminated (combining pure EMG and EMG dominant)
    signals.append(emg_eeg)
    signals.append(emg_eog_eeg)
    labels.extend([2] * (2 * samples_per_class))
    
    signals = np.vstack(signals)
    labels = np.array(labels)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(signals))
    signals = signals[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return signals, labels

def main():
    # Load original data
    EEG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EEG_all_epochs.npy')
    EOG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EOG_all_epochs.npy')
    EMG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EMG_all_epochs.npy')
    
    # Parameters
    samples_per_class = 500  # This will result in 500 clean, 1000 EOG, 1000 EMG samples
    
    # Generate mixed signals
    signals, labels = generate_mixed_signals(EEG_all, EOG_all, EMG_all, samples_per_class)
    
    # Save the data
    np.save('classification_signals.npy', signals)
    np.save('classification_labels.npy', labels)
    
    # Print statistics
    print(f"Generated {len(signals)} samples with shape {signals.shape}")
    print("Label distribution:")
    for i in range(3):
        print(f"Class {i}: {np.sum(labels == i)} samples")
    
    # Verify data independence
    unique_signals = len(np.unique(signals, axis=0))
    print(f"\nNumber of unique signals: {unique_signals}")
    print(f"Total number of signals: {len(signals)}")
    if unique_signals < len(signals):
        print("Warning: There are duplicate signals in the dataset!")

if __name__ == "__main__":
    main()