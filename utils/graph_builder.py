import torch
from torch_geometric.data import Data

def build_patch_graph(img_tensor, patch_size=16):
    _, H, W = img_tensor.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    patches = []
    for i in range(h_patches):
        for j in range(w_patches):
            patch = img_tensor[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.mean(dim=(1, 2)))  # Average RGB

    x = torch.stack(patches)  # Node features
    edge_index = []
    for i in range(len(patches)):
        for j in range(len(patches)):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).t().contiguous()

    return Data(x=x, edge_index=edge_index)
