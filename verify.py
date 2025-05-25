import torch
from torch_geometric.nn import GATConv

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.rand((4, 8))
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
gat = GATConv(8, 4)
out = gat(x, edge_index)
print("GATConv output shape:", out.shape)
