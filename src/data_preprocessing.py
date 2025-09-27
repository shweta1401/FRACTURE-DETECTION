"""
Data Preprocessing Script for MURA Dataset
- Converts images to grayscale
- Normalizes pixel values to [0,1]
- Resizes images to 224x224
- Saves preprocessed images to output directory
"""

import os
import cv2
import numpy as np


def normalize_image(image):
    """Scale pixel values to range [0,1]."""
    return image / 255.0


def resize_image(image, target_size=(224, 224)):
    """Resize an image to the given target size."""
    return cv2.resize(image, target_size)


def process_and_save_images(dataset_path, output_path):
    """
    Loops through dataset, preprocesses, and saves images.
    Args:
        dataset_path (str): Path to input dataset (train/valid).
        output_path (str): Path to save preprocessed dataset.
    """
    for body_part in os.listdir(dataset_path):  # e.g., XR_ELBOW, XR_FINGER
        part_path = os.path.join(dataset_path, body_part)
        if not os.path.isdir(part_path):
            continue

        for patient in os.listdir(part_path):  # e.g., patient00011
            patient_path = os.path.join(part_path, patient)
            for study in os.listdir(patient_path):  # e.g., study1_positive, study1_negative
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

                    # Normalize + resize
                    img = normalize_image(img)
                    img = resize_image(img)

                    # Save back as uint8
                    save_path = os.path.join(save_dir, img_file)
                    cv2.imwrite(save_path, (img * 255).astype(np.uint8))


if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA-v1.1"
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    VALID_PATH = os.path.join(DATASET_PATH, "valid")

    PROCESSED_PATH = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_processed"
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    process_and_save_images(TRAIN_PATH, os.path.join(PROCESSED_PATH, "train"))
    process_and_save_images(VALID_PATH, os.path.join(PROCESSED_PATH, "valid"))

    print("✅ All images processed and saved successfully!")
