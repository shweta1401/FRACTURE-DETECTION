"""
Create Patient-Level CSV for MURA Dataset
- Prevents data leakage by ensuring patient-wise splits
- Generates a CSV with image path, bone type, study, patient ID, fracture label
"""

import os
import pandas as pd


def create_patient_level_csv(all_data_dir, csv_out):
    """
    Traverse dataset and generate CSV with metadata.
    Args:
        all_data_dir (str): Path to unified dataset folder.
        csv_out (str): Path to save CSV file.
    """
    all_img_paths, bone_types, study_labels, patient_ids, fracture_status = [], [], [], [], []

    for bone in os.listdir(all_data_dir):
        bone_dir = os.path.join(all_data_dir, bone)
        if not os.path.isdir(bone_dir):
            continue

        for patient in os.listdir(bone_dir):
            patient_dir = os.path.join(bone_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for study in os.listdir(patient_dir):
                study_dir = os.path.join(patient_dir, study)
                if not os.path.isdir(study_dir):
                    continue

                # Label fracture presence from study folder name
                fracture = 1 if "positive" in study.lower() else 0

                for img in os.listdir(study_dir):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(study_dir, img)
                        all_img_paths.append(img_path)
                        bone_types.append(bone)
                        study_labels.append(study)
                        patient_ids.append(patient)
                        fracture_status.append(fracture)

    df = pd.DataFrame({
        "image_path": all_img_paths,
        "bone_type": bone_types,
        "study": study_labels,
        "patient_id": patient_ids,
        "fracture_label": fracture_status
    })

    # Sort for reproducibility
    df = df.sort_values(by=["bone_type", "patient_id", "study", "image_path"]).reset_index(drop=True)

    df.to_csv(csv_out, index=False)
    print("✅ CSV created:", csv_out)
    return df


if __name__ == "__main__":
    all_data_dir = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_all"
    csv_out = r"C:\Users\Shweta Sukhtankar\Desktop\PROJECT2\MURA_all_images.csv"

    df = create_patient_level_csv(all_data_dir, csv_out)
    print(df.head())
