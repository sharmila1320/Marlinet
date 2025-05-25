import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class MARLINet(nn.Module):
    def __init__(self, in_channels=3, gnn_channels=64):
        super(MARLINet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, gnn_channels, 3, padding=1),
            nn.ReLU()
        )
        self.gnn = GATConv(gnn_channels, gnn_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(gnn_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

def forward(self, img, node_feats, edge_index):
    x = self.encoder(img)
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1).permute(0, 2, 1).squeeze(0)  # [N, C]
    print(f"[GNN DEBUG] x_flat: {x_flat.shape}")
    x_gnn = self.gnn(x_flat, edge_index)
    print(f"[GNN DEBUG] x_gnn: {x_gnn.shape}")
    x_out = x_gnn.permute(1, 0).view(1, C, H, W)
    print(f"[GNN DEBUG] x_out: {x_out.shape}")
    out = self.decoder(x_out)
    return out
