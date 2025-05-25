import os
from glob import glob

input_folder = 'datasets/UIEB/input'
extensions = ['*.PNG']

image_paths = []
for ext in extensions:
    image_paths.extend(glob(os.path.join(input_folder, ext)))

print(f"Found {len(image_paths)} PNG files:")
for path in image_paths[:5]:
    print(path)
