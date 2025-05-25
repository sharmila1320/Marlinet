import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from models.marlinet import MARLINet
from utils.graph_builder import build_patch_graph

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MARLINet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

input_dir = 'datasets/UIEB/degraded'
gt_dir = 'datasets/UIEB/gt'

input_images = sorted(os.listdir(input_dir))
gt_images = sorted(os.listdir(gt_dir))

for epoch in range(5):  # You can increase this later
    total_loss = 0
    for img_name in input_images:
        if img_name not in gt_images:
            continue

        input_path = os.path.join(input_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)

        try:
            input_img = Image.open(input_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to open {img_name}: {e}")
            continue

        input_tensor = transform(input_img).to(device)
        gt_tensor = transform(gt_img).to(device)
        print(f"[DEBUG] Input max: {input_tensor.max():.4f}, GT max: {gt_tensor.max():.4f}")

        data = build_patch_graph(input_tensor.cpu())
        node_feats = data.x.to(device)
        edge_index = data.edge_index.to(device)
        print(f"[DEBUG] node_feats shape: {node_feats.shape}")
        print(f"[DEBUG] edge_index shape: {edge_index.shape}")
        print(f"[DEBUG] input_tensor shape: {input_tensor.shape}")

        output = model(input_tensor.unsqueeze(0), node_feats, edge_index)
        loss = criterion(output, gt_tensor.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(input_images):.4f}")
