import torch.nn as nn
from torch_geometric.nn import GIN, GAT, GATv2Conv
import torch

# This may be better because LSTM will prevent overfitting and instead of context-based spatiotemporal fusion,
# LSTM temporal features are recomputed with graph attention and temporal transforms are done by learning the structure of the latent dimension (GIN)
class NeuroTransform(nn.Module):
    def __init__(self, n_emg_timesteps):
        super().__init__()
        
        self.lstm_encoder = nn.LSTM(input_size=n_emg_timesteps, hidden_size=256, num_layers=1)
        
        self.node_wise_attn = GATv2Conv(in_channels=10, out_channels=32)
        self.channel_attn = GAT(
            in_channels=256,
            hidden_channels=64,
            out_channels=32,
            num_layers=4
        )
        
        self.channel_transform = GAT(
            in_channels=5,
            hidden_channels=16,
            out_channels=32,
            num_layers=2
        )
        
        self.temporal_signal_reconstruction = GIN(
            in_channels=32,
            hidden_channels=256,
            out_channels=n_emg_timesteps,
            num_layers=4
        )

        
    def forward(self, emg_graph):
        lstm_encoded = self.lstm_encoder(emg_graph.x.unsqueeze(0))
        output, (hn, cn) = lstm_encoded
        
        gat_out, node_attn = self.node_wise_attn(emg_graph.x, emg_graph.edge_index, return_attention_weights=True)
        idx, weights = node_attn
        gat_encoded_latent = self.channel_attn(output.squeeze(0), emg_graph.edge_index, emg_graph.edge_attr * weights)
                
        feature_edge_index = torch.combinations(torch.arange(32), r=2, with_replacement=False).T
        edge_index = torch.cat([feature_edge_index, feature_edge_index.flip(0)], dim=1)
        
        # print(gat_encoded_latent.permute(1, 0).shape)
        # print(edge_index.shape)
        channel_transformed = self.channel_transform(gat_encoded_latent.permute(1, 0), edge_index).permute(1, 0)
        
        out_eeg_signal = self.temporal_signal_reconstruction(channel_transformed, emg_graph.edge_index)
        # print(out_eeg_signal.shape)
        return out_eeg_signal