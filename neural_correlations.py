import numpy as np
import pandas as pd
from neural_data_graph_construction import compute_adj_matrix_eeg_mat
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Findings: Occipital lobe show high correlations with each other as well as frontal lobe. Temporal lobe, not so much and this is justifiered
# because it is not directly correlated with motor.

# Execute: python neural_correlations.py
# eeg_data = np.load('eeg_signals_copy_test.npy').reshape(4032, 32, 1200)
# random_idx = random.randint(0, 4000)
# sample_eeg_data = eeg_data[random_idx]
# signal, adj_mtx_test = compute_adj_matrix_eeg_mat(sample_eeg_data, sampling_frequency=500)
channel_names = np.array([
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ])
def get_corr_df(adj_matrix):
    

    adj_df = pd.DataFrame(adj_matrix, columns=channel_names, index=channel_names)

    correlations = []
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            correlations.append({
                "Channel 1": channel_names[i],
                "Channel 2": channel_names[j],
                "Correlation": adj_df.iloc[i, j]
            })

    correlation_df = pd.DataFrame(correlations)

    # Sort (optional)
    correlation_df = correlation_df.sort_values(by="Correlation", ascending=False)
    return correlation_df
    # correlation_df.to_csv("channel_pair_correlations.csv", index=False)

# print(correlation_df.head())

# plt.figure(figsize=(12, 10))
# sns.heatmap(adj_mtx_test, xticklabels=channel_names, yticklabels=channel_names, cmap="coolwarm", annot=False)
# plt.title("EEG Channel Pairwise Correlations")
# plt.show()