import torch.nn as nn
from torch_geometric.nn import GIN, GAT
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, emg_tsteps, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.embed = nn.Linear(emg_tsteps, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure we are using the correct sequence length dimension of x
        x = self.embed(x)
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x).squeeze(0)
    
class NeuroTransform(nn.Module):
    def __init__(self, n_emg_timesteps):
        super().__init__()
        self.pos_encoder_initial = PositionalEmbedding(emg_tsteps=n_emg_timesteps, d_model=128)
        
        temp_enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_temp_enc = nn.TransformerEncoder(temp_enc_layer, num_layers=2)
        
        self.graph_encoder = GIN(
            in_channels=128,
            hidden_channels=256,
            out_channels=512,
            num_layers=2
        )
        
        # Optional: No concrete evidence it'll work but maybe it might experimentally
        self.graph_attention_bottleneck = GAT(
            in_channels=512,
            hidden_channels=512,
            out_channels=512,
            num_layers=1
        )
        
        self.graph_channel_decoder = GIN(
            in_channels=5,
            hidden_channels=16,
            out_channels=32,
            num_layers=2
        )
        
        self.pos_encoder_feature_dec = PositionalEmbedding(emg_tsteps=512, d_model=512)  # (32, 512)

        # Technically a transformer encoder layer, but its a "decoder" for this model
        temp_dec_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.transformer_temp_dec = nn.TransformerEncoder(temp_dec_layer, num_layers=2)
        
        # Output is to n_EMG_timesteps because its the same as EEG
        self.out_transform = nn.Linear(512, n_emg_timesteps)

        
    def forward(self, emg_graph):
        pos_encoded_emg_signal = self.pos_encoder_initial(emg_graph.x)
        temp_encoded_emg = self.transformer_temp_enc(pos_encoded_emg_signal)
        
        spatial_features = self.graph_encoder(temp_encoded_emg, emg_graph.edge_index, emg_graph.edge_attr)
        attn_bneck = self.graph_attention_bottleneck(spatial_features, emg_graph.edge_index, emg_graph.edge_attr)
        dec_graph_features = self.graph_channel_decoder(attn_bneck.T, emg_graph.edge_index, emg_graph.edge_attr)
        
        pos_enc_features = self.pos_encoder_feature_dec(dec_graph_features.T)
        temp_decoded = self.transformer_temp_dec(pos_enc_features)
        out_signal = self.out_transform(temp_decoded)
        
        return out_signal