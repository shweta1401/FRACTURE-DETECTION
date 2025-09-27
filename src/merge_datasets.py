"""
Merge Processed and Augmented MURA Dataset
- Combines original preprocessed and augmented datasets into one unified folder
"""

import os
import shutil


def merge_to_all(source, target_root):
    """
    Merge images from a source directory into a unified destination.
    Args:
        source (str): Path to source dataset (train/valid, processed/augmented).
        target_root (str): Path to unified dataset directory.
    """
    for root, _, files in os.walk(source):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            rel_path = os.path.relpath(root, source)  # relative subpath
            dest_path = os.path.join(target_root, rel_path)

            os.makedirs(dest_path, exist_ok=True)

            src_file = os.path.join(root, file)
            tgt_file = os.path.join(dest_path, file)

            # Avoid duplicates
            if not os.path.exists(tgt_file):
                shutil.copy2(src_file, tgt_file)


if __name__ == "__main__":
    # Example paths (adapt for your system)
    processed_train = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_processed\train"
    processed_valid = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_processed\valid"
    augmented_train = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_augmented1\train"
    augmented_valid = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_augmented1\valid"

    all_data_dir = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_all"
    os.makedirs(all_data_dir, exist_ok=True)

    # Merge all datasets
    merge_to_all(processed_train, all_data_dir)
    merge_to_all(processed_valid, all_data_dir)
    merge_to_all(augmented_train, all_data_dir)
    merge_to_all(augmented_valid, all_data_dir)

    print("✅ All images combined into:", all_data_dir)
