# Obtain EMG/EEG signals (call it "neural_data") and calculate statistical similarity for weighted edges
# Construct graph with ROIs (channels) and a connectivity matrix, weighted by statistical similarity
# Compute and return adjacency matrix to input to a GNN

import numpy as np
from scipy.signal import welch, hilbert
from scipy.fft import fft
import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt

def compute_plv(signal1, signal2):
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)
    
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)
    
    phase_diff = phase1 - phase2
    
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))  # Mean of the complex exponentials of the phase difference
    
    return plv

def compute_plv_adjacency_matrix(eeg_data):
    n_channels = eeg_data.shape[0] 
    adjacency_matrix = np.zeros((n_channels, n_channels)) 
    for i in range(n_channels):
        for j in range(i + 1, n_channels):  
            plv_value = 1 - compute_plv(eeg_data[i, :], eeg_data[j, :])
            adjacency_matrix[i, j] = plv_value
            adjacency_matrix[j, i] = plv_value  
            
    np.fill_diagonal(adjacency_matrix, 1)
    return adjacency_matrix



def compute_adj_matrix_neural_mat(neural_signal, sampling_freq):
    dist_muscle_matrix = np.array([
        [0, 17.5, 27.5, 27.5, 32.5],  # Anterior Deltoid
        [17.5, 0, 7.5, 12.5, 17.5],  # Brachioradialis
        [27.5, 7.5, 0, 2, 5],  # Flexor Digitorum
        [27.5, 12.5, 2, 0, 3.5],  # Common Extensor Digitorum
        [32.5, 17.5, 5, 3.5, 0]  # First Dorsal Interosseus
    ])

    dist_adj_matrix = np.abs(np.corrcoef(dist_muscle_matrix))

    psd_data = []
    for channel_data in neural_signal:
        freqs, psd = welch(channel_data, fs=sampling_freq, nperseg=500)
        psd_data.append(psd)
    psd_data = np.array(psd_data)
    psd_corr_matrix = np.abs(np.corrcoef(psd_data))

    epsilon = np.percentile(dist_muscle_matrix, 95)  # Epsilon is 95th percentile of the data
    adjacency_matrix = np.exp(-dist_muscle_matrix ** 2 / epsilon)
    # print(adjacency_matrix)

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    lp_corr_matrix = np.corrcoef(laplacian_matrix)
    # print(lp_corr_matrix)

    total_emg_adj_matrix = (dist_adj_matrix + psd_corr_matrix + np.abs(lp_corr_matrix)) / 3

    return total_emg_adj_matrix

def compute_adj_matrix_eeg_mat(eeg_signal, sampling_frequency):
    """
    Node correlation weighting:
     - Laplacian Filtering (spatial),
     - Frequency Fourier transformation (frequency),
     - Power spectral density (temporal) - keep because it measures signal strength over correlation frequency bands
     - Euclidian Distance (spatial)

    The correlation matrices generated for each similarity function are averaged to compute a final one

    """

    # print(eeg_signal)

    # Computes signal with respect to the signals of neighbors (second spatial derivative)
    pos = np.array([[-2.7, 8.6, 3.6],
                    [2.7, 8.6, 3.6],
                    [-6.7, 5.2, 3.6],
                    [-4.7, 6.2, 8.],
                    [0., 6.7, 9.5],
                    [4.7, 6.2, 8.],
                    [6.7, 5.2, 3.6],
                    [-5.5, 3.2, 6.6],
                    [-3., 3.3, 11.],
                    [3., 3.3, 11.],
                    [5.5, 3.2, 6.6],
                    [-7.8, 0., 3.6],
                    [-6.1, 0., 9.7],
                    [0., 0., 12.],
                    [6.1, 0., 9.7],
                    [7.8, 0., 3.6],
                    [-7.3, -2.5, 0.],
                    [-7.2, -2.7, 6.6],
                    [-3., -3.2, 11.],
                    [3., -3.2, 11.],
                    [7.2, -2.7, 6.6],
                    [7.3, -2.5, 0.],
                    [-6.7, -5.2, 3.6],
                    [-4.7, -6.2, 8.],
                    [0., -6.7, 9.5],
                    [4.7, -6.2, 8.],
                    [6.7, -5.2, 3.6],
                    [-4.7, -6.7, 0.],
                    [-2.7, -8.6, 3.6],
                    [0., -9., 3.6],
                    [2.7, -8.6, 3.6],
                    [4.7, -6.7, 0.]])
    distance_matrix = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
    # print(distance_matrix)
    distance_corr_matrix = np.corrcoef(distance_matrix)

    # print(distance_corr_matrix)

    epsilon = np.percentile(distance_matrix, 95)  # Epsilon is 95th percentile of the data
    adjacency_matrix = np.exp(-distance_matrix ** 2 / epsilon)

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    lp_corr_matrix = np.corrcoef(laplacian_matrix)

    # print(lp_corr_matrix)

    corr_matrix_freq = np.corrcoef(np.abs(fft(eeg_signal)))
    # print(corr_matrix_freq)

    psd_data = []
    for channel_data in eeg_signal:
        freqs, psd = welch(channel_data, fs=sampling_frequency, nperseg=512)
        psd_data.append(psd)
    psd_data = np.array(psd_data)
    psd_corr_matrix = np.corrcoef(psd_data)
    # print(psd_corr_matrix.shape)
    
    plv_corr_matrix = compute_plv_adjacency_matrix(eeg_signal)
    

    neural_adjacency_matrix = (np.abs(psd_corr_matrix) + np.abs(corr_matrix_freq) + np.abs(
        distance_corr_matrix) + np.abs(lp_corr_matrix)) / 4
    
    n_ajd_mtx_simpler = (np.abs(plv_corr_matrix) + np.abs(distance_corr_matrix)) / 2

    return eeg_signal, n_ajd_mtx_simpler


def compute_adj_matrix_emg_mat(emg_signal, sampling_frequency):
    # print(emg_signal)

    dist_muscle_matrix = np.array([
        [0, 17.5, 27.5, 27.5, 32.5],  # Anterior Deltoid
        [17.5, 0, 7.5, 12.5, 17.5],  # Brachioradialis
        [27.5, 7.5, 0, 2, 5],  # Flexor Digitorum
        [27.5, 12.5, 2, 0, 3.5],  # Common Extensor Digitorum
        [32.5, 17.5, 5, 3.5, 0]  # First Dorsal Interosseus
    ])

    dist_adj_matrix = np.abs(np.corrcoef(dist_muscle_matrix))
    # print(dist_adj_matrix)

    psd_data = []
    for channel_data in emg_signal:
        freqs, psd = welch(channel_data, fs=sampling_frequency, nperseg=500)
        psd_data.append(psd)
    psd_data = np.array(psd_data)
    psd_corr_matrix = np.abs(np.corrcoef(psd_data))

    corr_matrix_freq = np.corrcoef(np.abs(fft(emg_signal)))

    total_emg_adj_matrix = (dist_adj_matrix + psd_corr_matrix + corr_matrix_freq) / 3

    return emg_signal, total_emg_adj_matrix

