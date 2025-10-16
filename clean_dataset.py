import os
import shutil
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
dataset_dir = r"C:\Users\Acer\Documents\LargeDS\LargeDS"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")
rejected_dir = os.path.join(dataset_dir, "rejected")

# Updated 22 tree species (removed banyan, Desert Date, jamun)
expected_classes = [
    'amla', 'Anjan', 'chir_pine', 'coconut', 'Guava',
    'indian_beech', 'indian_trumpet', 'jackfruit', 'Jand', 'Karungali',
    'Mango', 'muringa_tree', 'neem_tree', 'palmyra_palm', 'peepal', 'punnai',
    'sandalwood', 'Teak', 'turmeric_tree', 'Vagai', 'Vathakkani', 'wild_date_palm'
]

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
MIN_IMAGES_PER_CLASS = 8  # Lowered for Mango
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}

def create_directories():
    """Create train, val, test, and rejected directories."""
    for dir_path in [train_dir, val_dir, test_dir, rejected_dir]:
        Path(dir_path).mkdir(exist_ok=True)
        for cls in expected_classes:
            Path(os.path.join(dir_path, cls)).mkdir(exist_ok=True)
    logger.info("Created directory structure")

def is_valid_image(file_path):
    """Check if an image is valid and readable."""
    try:
        img = Image.open(file_path)
        img.verify()
        img.close()
        return True
    except Exception as e:
        logger.error(f"Invalid image {file_path}: {e}")
        return False

def convert_to_rgb(file_path, output_path):
    """Convert image to RGB and resize to 224x224."""
    try:
        img = Image.open(file_path)
        if img.mode != 'RGB':
            logger.info(f"Converting {file_path} to RGB")
            img = img.convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
        img.save(output_path, quality=95)
        img.close()
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def infer_class_from_filename(filename):
    """Infer class from file name."""
    filename_lower = filename.lower()
    for cls in expected_classes:
        if cls.lower().replace(' ', '_') in filename_lower:
            return cls
    return 'unknown'

def collect_images(dataset_dir):
    """Collect all valid images and their class labels."""
    images = []
    for root, _, files in os.walk(dataset_dir):
        if root in [train_dir, val_dir, test_dir, rejected_dir]:
            continue  # Skip already processed splits
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                if is_valid_image(file_path):
                    # Infer class from file name
                    class_name = infer_class_from_filename(file)
                    if class_name == 'unknown':
                        # Try folder name as fallback
                        folder_name = os.path.basename(os.path.dirname(file_path))
                        if folder_name in expected_classes:
                            class_name = folder_name
                    images.append((file_path, class_name))
                else:
                    shutil.move(file_path, os.path.join(rejected_dir, file))
    logger.info(f"Collected {len(images)} valid images")
    return images

def balance_and_split(images):
    """Balance classes, rename files, and split into train, val, test."""
    class_images = {cls: [] for cls in expected_classes}
    class_images['unknown'] = []

    # Group images by class
    for file_path, class_name in images:
        class_images[class_name].append((file_path, class_name))

    # Report initial class counts
    logger.info("Initial class counts:")
    for cls, imgs in class_images.items():
        logger.info(f"{cls}: {len(imgs)} images")

    # Ensure minimum images per class
    for cls in expected_classes:
        if len(class_images.get(cls, [])) < MIN_IMAGES_PER_CLASS:
            logger.warning(f"{cls} has {len(class_images.get(cls, []))} images, below minimum {MIN_IMAGES_PER_CLASS}")
            current_images = class_images.get(cls, [])
            while len(current_images) < MIN_IMAGES_PER_CLASS and current_images:
                for img in current_images[:MIN_IMAGES_PER_CLASS - len(current_images)]:
                    current_images.append(img)
            class_images[cls] = current_images

    # Split images and rename
    for cls in expected_classes:
        imgs = class_images.get(cls, [])
        if not imgs:
            logger.error(f"No images for class {cls}. Training may fail.")
            continue
        np.random.shuffle(imgs)
        total = len(imgs)
        train_end = int(total * SPLIT_RATIOS['train'])
        val_end = train_end + int(total * SPLIT_RATIOS['val'])

        # Rename and move
        for i, (img_path, _) in enumerate(imgs):
            ext = os.path.splitext(img_path)[1]
            if i < train_end:
                split_dir = train_dir
            elif i < val_end:
                split_dir = val_dir
            else:
                split_dir = test_dir
            new_filename = f"{cls}_{i:04d}{ext}"
            output_path = os.path.join(split_dir, cls, new_filename)
            if convert_to_rgb(img_path, output_path):
                logger.info(f"Renamed and moved {img_path} to {output_path}")
            else:
                shutil.move(img_path, os.path.join(rejected_dir, os.path.basename(img_path)))

    # Handle unknown class
    for i, (img_path, _) in enumerate(class_images['unknown']):
        ext = os.path.splitext(img_path)[1]
        new_filename = f"unknown_{i:04d}{ext}"
        output_path = os.path.join(rejected_dir, new_filename)
        shutil.move(img_path, output_path)
        logger.info(f"Moved unknown class image to {output_path}")

def generate_dataset_report():
    """Generate a report of class counts and dataset statistics."""
    report = {"train": {}, "val": {}, "test": {}, "rejected": {}}
    for split, dir_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for cls in expected_classes:
            cls_dir = os.path.join(dir_path, cls)
            report[split][cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))])
        logger.info(f"{split.capitalize()} class counts: {report[split]}")
    report["rejected"]["count"] = len([f for f in os.listdir(rejected_dir) if f.lower().endswith(('.jpg', '.png'))])
    logger.info(f"Rejected images: {report['rejected']['count']}")
    with open(os.path.join(dataset_dir, "dataset_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    logger.info("Dataset report saved to dataset_report.json")

def main():
    logger.info("Starting data cleaning process")
    create_directories()
    images = collect_images(dataset_dir)
    balance_and_split(images)
    generate_dataset_report()
    logger.info("Data cleaning complete!")

if __name__ == "__main__":
    main()