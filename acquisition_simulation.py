import os
from glob import glob
from PIL import Image, ImageFilter

def degrade_image_pil(img):
    """
    Apply degradation: grayscale + Gaussian blur
    """
    gray = img.convert('L')  # Convert to grayscale
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))  # Apply blur
    return blurred

def process_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Collect all image paths with supported extensions
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"[INFO] Found {len(image_paths)} image files to process.")

    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing: {path}")
        try:
            img = Image.open(path).convert('RGB')  # Open + ensure RGB format
            print("[INFO] Image opened with PIL")

            degraded = degrade_image_pil(img)
            print("[INFO] Image degraded")

            # Save as .jpg with same base name
            base = os.path.splitext(os.path.basename(path))[0]
            name = base + '.jpg'
            save_path = os.path.join(output_folder, name)

            degraded.save(save_path, 'JPEG')
            print(f"[OK] Saved to: {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

# Run the script with absolute paths
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, 'datasets', 'UIEB', 'input')
    output_folder = os.path.join(base_dir, 'datasets', 'UIEB', 'degraded')

    print(f"\n[INFO] Reading from: {input_folder}")
    print(f"[INFO] Saving to: {output_folder}")
    process_dataset(input_folder, output_folder)
    print("[DONE] Degradation complete.\n")
