import os
import glob
import nibabel as nib
import numpy as np

def compute_angular_stats(ref_path, pred_path, std_factor=3.0):
    """
    Compute angular difference (in degrees) between reference and predicted 
    directions voxel by voxel for one pair of NIfTI files, then derive:
      - number of outliers (> mean + std_factor * std)
      - total number of valid (non-zero) voxels
      - percentage of outliers
      - largest angular difference

    Returns a dictionary with these metrics.
    """
    # Load NIfTI
    ref_img = nib.load(ref_path)
    pred_img = nib.load(pred_path)
    
    ref_data = ref_img.get_fdata()  # shape: (X, Y, Z, 3) typically
    pred_data = pred_img.get_fdata()  # shape: (X, Y, Z, 3) typically

    if ref_data.shape != pred_data.shape:
        raise ValueError(f"Shape mismatch:\n  {ref_path}: {ref_data.shape}\n  {pred_path}: {pred_data.shape}")

    # Reshape to (N, 3)
    shape_3d = ref_data.shape[:3]
    ref_vectors = ref_data.reshape(-1, 3)
    pred_vectors = pred_data.reshape(-1, 3)

    # Mask of valid voxels: norm != 0
    ref_norms = np.linalg.norm(ref_vectors, axis=1)
    pred_norms = np.linalg.norm(pred_vectors, axis=1)
    valid_mask = (ref_norms != 0) & (pred_norms != 0)

    num_valid_voxels = np.sum(valid_mask)
    if num_valid_voxels == 0:
        # Edge case: No valid voxels
        return {
            'num_outliers': 0,
            'total_valid_voxels': 0,
            'pct_outliers': 0.0,
            'max_diff_value': 0.0
        }

    ref_valid = ref_vectors[valid_mask]
    pred_valid = pred_vectors[valid_mask]

    # Compute angular differences
    dot_product = np.sum(ref_valid * pred_valid, axis=1)
    norms_product = (np.linalg.norm(ref_valid, axis=1) *
                     np.linalg.norm(pred_valid, axis=1))
    
    cos_angle = dot_product / norms_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles_rad = np.arccos(cos_angle)
    angles_deg = np.degrees(angles_rad)
    
    # Mean, std, threshold
    mean_angle = np.mean(angles_deg)
    std_angle = np.std(angles_deg)
    threshold = mean_angle + std_factor * std_angle

    # Outliers
    outlier_mask = (angles_deg > threshold)
    num_outliers = np.sum(outlier_mask)

    # Largest angular difference
    max_angle = np.max(angles_deg) if angles_deg.size > 0 else 0.0

    # Percentage
    pct_outliers = (num_outliers / num_valid_voxels) * 100.0
    
    results = {
        'num_outliers': int(num_outliers),
        'total_valid_voxels': int(num_valid_voxels),
        'pct_outliers': float(pct_outliers),
        'max_diff_value': float(max_angle)
    }
    return results


def compare_all_tracts(ref_dir, pred_dir, out_report="summary_report.txt", std_factor=3.0):
    """
    1. Finds all reference .nii.gz files in ref_dir that do NOT have '_v1' in the name.
    2. For each, locates the predicted file with the same base name + '_v1.nii.gz' in pred_dir.
    3. Computes stats and writes a summary line to out_report.
    
    The summary line contains:
        TractName, NumOutliers, TotalValidVoxels, PctOutliers, LargestAngularDiff
    """
    # Grab all .nii.gz in ref_dir that do NOT end with '_v1.nii.gz'
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.nii.gz")))
    ref_files = [f for f in ref_files if not f.endswith("_dominant.nii.gz")]

    with open(out_report, "w") as f:
        f.write("Tract Comparison Summary (Outliers > mean + 3*std)\n")
        f.write("-------------------------------------------------\n")
        f.write("TractName, NumOutliers, TotalValidVoxels, PctOutliers, LargestAngularDiff\n")

        for ref_path in ref_files:
            base_name = os.path.basename(ref_path)  # e.g. "TractName.nii.gz"

            # Remove .nii.gz (if present) to get the "TractName"
            if base_name.endswith(".nii.gz"):
                base_name_noext = base_name[:-7]  # remove ".nii.gz"
            else:
                base_name_noext = os.path.splitext(base_name)[0]

            # Predicted file is base_name_noext + "_v1.nii.gz" in pred_dir
            pred_name = base_name_noext + "_dominant.nii.gz"
            pred_path = os.path.join(pred_dir, pred_name)

            if not os.path.exists(pred_path):
                # If predicted file not found, skip or warn
                print(f"WARNING: No predicted file found for '{ref_path}' "
                      f"at '{pred_path}'. Skipping.")
                continue
            
            # Compute stats
            stats = compute_angular_stats(ref_path, pred_path, std_factor=std_factor)

            # Write a summary row
            f.write(f"{base_name_noext}, "
                    f"{stats['num_outliers']}, "
                    f"{stats['total_valid_voxels']}, "
                    f"{stats['pct_outliers']:.2f}, "
                    f"{stats['max_diff_value']:.4f}\n")


if __name__ == "__main__":
   
    ref_dir = "/home/s65eelma/TOM"  
    pred_dir = "/home/s65eelma/Dominant_new25" 
    output_file = "new_summary_report25.txt"

    compare_all_tracts(ref_dir, pred_dir, out_report=output_file, std_factor=3.0)
    print(f"Comparison complete. Summary written to '{output_file}'.")
