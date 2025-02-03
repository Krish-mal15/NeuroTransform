import torch
from torch_geometric.data import Data
from graph_data_preprocessing import emg_test_data, edge_index_emg_test, edge_weight_emg_test
from neurotransform_main_v2 import NeuroTransform
from graph_data_preprocessing import inverse_z_score_norm

from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from mne import create_info

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
eeg_recon_model = NeuroTransform(n_emg_timesteps=10).to(device)
eeg_recon_model.load_state_dict(torch.load("models/neurotransform_main_new_v2_NEWEST", weights_only=True, map_location=device))


def calculate_nrmse(true_signal, predicted_signal):
    true_signal = true_signal.flatten()
    predicted_signal = predicted_signal.flatten()

    rmse = torch.sqrt(torch.mean((true_signal - predicted_signal) ** 2))

    norm = torch.max(true_signal) - torch.min(true_signal)

    nrmse = rmse / norm
    return nrmse.item()

def calculate_similarity(true_signal, predicted_signal):
    nrmse = calculate_nrmse(true_signal, predicted_signal)
    
    similarity_percentage = (1 - nrmse) * 100
    return similarity_percentage

def seg_based_accuracy():
    total_similarity = 0
    count = 0
    num_test_samples = 500
    sample_counter = 0

    eeg_recon_model.eval()
    
    emg_graph_data = Data(x=torch.tensor(emg_test_data[:, :10], dtype=torch.float), edge_attr=edge_weight_emg_test, edge_index=edge_index_emg_test)
    with torch.no_grad():
        print(emg_graph_data)
        eeg_decoded_pred = eeg_recon_model(emg_graph_data).numpy()
        print(eeg_decoded_pred.shape)
        
        mean_signal = eeg_decoded_pred.mean(axis=1)
        channel_names = [
                    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8',
                    'O1', 'Oz', 'O2', 'AF7', 'AF8', 'FC5', 'FC1', 'FC2', 'FC6',
                    'CP5', 'CP1', 'CP2', 'CP6', 'PO7', 'PO8'
                ]
        montage = make_standard_montage('standard_1020')
        info = create_info(ch_names=channel_names, sfreq=500, ch_types='eeg')
        info.set_montage(montage)

        # Get channel positions
        pos = np.array([montage.get_positions()['ch_pos'][ch][:2] for ch in channel_names])

        # Plot the topomap
        fig, ax = plt.subplots()
        im, _ = plot_topomap(mean_signal, pos, axes=ax, show=False, cmap='RdBu_r')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Signal Pred Value')
        plt.title('Mean EEG Signal Pred Topomap')
        plt.show()
    
    
    
    # with torch.no_grad():
    #     for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg in dataloader:
    #         if sample_counter > num_test_samples:
    #             break
    #         features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg = features_emg.to(device), edge_index_emg.to(device), edge_weight_emg.to(device), features_eeg.to(device), edge_index_eeg.to(device), edge_weight_eeg.to(device)
    #         emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
            
    #         eeg_true_mean = torch.mean(features_eeg, axis=1, keepdims=True)
    #         eeg_true_std_dev = torch.std(features_eeg, axis=1, keepdims=True)
            
            
    #         eeg_decoded_pred = np.array(inverse_z_score_norm(eeg_recon_model(emg_graph_data), eeg_true_mean, eeg_true_std_dev).squeeze(0))
    #         print('')
    #         print("PRED: ", eeg_decoded_pred.shape)
    #         print("TRUE: ", features_eeg.shape)
    #         print('')
            
    #         features_eeg = np.array(inverse_z_score_norm(features_eeg.squeeze(0), eeg_true_mean, eeg_true_std_dev).squeeze(0))
    #         # print(features_eeg.shape)
         
    #         channel_n_pred = eeg_decoded_pred[16]
    #         channel_n_true = features_eeg[16]
            
    #         # plt.figure(figsize=(8, 5))
    #         # plt.plot(channel_n_pred, marker='o', label='Pred', color='blue')
    #         # plt.plot(channel_n_true, marker='s', label='True', color='orange')

    #         # # Add labels, legend, and title
    #         # plt.xlabel('Time')
    #         # plt.ylabel('Amplitude')
    #         # plt.title('EEG Signals Comparison')
    #         # plt.legend()
    #         # plt.grid(True)
    #         # plt.show()
    #         mean_signal = eeg_decoded_pred.mean(axis=1)
    #         mean_sig_true = features_eeg.mean(axis=1)
    #         # print(mean_sig_true.shape)
            
    #         mean_sig_diff = mean_signal - mean_sig_true

    #         # Create MNE info object with standard 10-20 montage
    #         channel_names = [
    #             'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    #             'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8',
    #             'O1', 'Oz', 'O2', 'AF7', 'AF8', 'FC5', 'FC1', 'FC2', 'FC6',
    #             'CP5', 'CP1', 'CP2', 'CP6', 'PO7', 'PO8'
    #         ]
    #         montage = make_standard_montage('standard_1020')
    #         info = create_info(ch_names=channel_names, sfreq=500, ch_types='eeg')
    #         info.set_montage(montage)

    #         # Get channel positions
    #         pos = np.array([montage.get_positions()['ch_pos'][ch][:2] for ch in channel_names])

    #         # Plot the topomap
    #         fig, ax = plt.subplots()
    #         im, _ = plot_topomap(mean_signal, pos, axes=ax, show=False, cmap='RdBu_r')
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label('Mean Signal Pred Value')
    #         plt.title('Mean EEG Signal Pred Topomap')
    #         plt.show()
            
    #         fig, ax = plt.subplots()
    #         im, _ = plot_topomap(mean_sig_true, pos, axes=ax, show=False, cmap='RdBu_r')
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label('Mean Signal True Value')
    #         plt.title('Mean EEG Signal True Topomap')
    #         plt.show()
            
    #         fig, ax = plt.subplots()
    #         im, _ = plot_topomap(mean_sig_diff, pos, axes=ax, show=False, cmap='RdBu_r')
    #         cbar = plt.colorbar(im, ax=ax)
    #         cbar.set_label('Mean Signal Diff Value')
    #         plt.title('Mean EEG Signal Diff Topomap')
    #         plt.show()

        #     similarity = calculate_similarity(inverse_z_score_norm(features_eeg, eeg_true_mean, eeg_true_std_dev), eeg_decoded_pred)
        #     print(similarity)
        #     total_similarity += similarity
        #     count += 1
        #     sample_counter += 1
        #     print(sample_counter)
                
        # avg_similarity = total_similarity / count
        # print(avg_similarity)
        
seg_based_accuracy()
