"""
Data Augmentation Script for Preprocessed MURA Dataset
- Applies augmentations (flip, rotation, brightness, blur, resize)
- Saves augmented images into output directory
"""

import os
import cv2
import numpy as np
import albumentations as A


# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
    A.GaussianBlur(p=0.2),
    A.Resize(height=224, width=224)
])


def augment_and_save_images(dataset_path, output_path):
    """
    Apply augmentation and save augmented images.
    Args:
        dataset_path (str): Path to preprocessed dataset (train/valid).
        output_path (str): Path to save augmented dataset.
    """
    for body_part in os.listdir(dataset_path):
        part_path = os.path.join(dataset_path, body_part)
        if not os.path.isdir(part_path):
            continue

        for patient in os.listdir(part_path):
            patient_path = os.path.join(part_path, patient)
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)

                save_dir = os.path.join(output_path, body_part, patient, study)
                os.makedirs(save_dir, exist_ok=True)

                for img_file in os.listdir(study_path):
                    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    img_path = os.path.join(study_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f"⚠️ Warning: Could not load {img_path}")
                        continue

                    # Apply augmentation
                    augmented_image = augmentations(image=img)["image"]

                    # Normalize [0-1], then convert to uint8 for saving
                    augmented_image = (augmented_image / 255.0 * 255).astype(np.uint8)

                    save_path = os.path.join(save_dir, f"aug_{img_file}")
                    cv2.imwrite(save_path, augmented_image)


if __name__ == "__main__":
    PROCESSED_PATH = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_processed"
    TRAIN_PATH = os.path.join(PROCESSED_PATH, "train")
    VALID_PATH = os.path.join(PROCESSED_PATH, "valid")

    AUGMENTED_PATH = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_augmented1"
    os.makedirs(AUGMENTED_PATH, exist_ok=True)

    augment_and_save_images(TRAIN_PATH, os.path.join(AUGMENTED_PATH, "train"))
    augment_and_save_images(VALID_PATH, os.path.join(AUGMENTED_PATH, "valid"))

    print("✅ All images successfully augmented and saved!")
