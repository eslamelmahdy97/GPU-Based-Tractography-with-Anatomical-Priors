import os
import numpy as np
import nibabel as nib

# -------------------------------------------------
# Directories and file name setup
# -------------------------------------------------
reference_dir = "/home/s65eelma/TOM"
direction_dir = "/home/s65eelma/New_output"
output_dir    = "Dominant_new25"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# Go through each reference file
# -------------------------------------------------
for ref_file in os.listdir(reference_dir):
    if not ref_file.endswith(".nii.gz"):
        continue

    # Extract tract name from filename, e.g. "CC" from "CC.nii.gz"
    tract_name = ref_file.replace(".nii.gz", "")
    ref_path   = os.path.join(reference_dir, ref_file)

    # Load reference 
    ref_img  = nib.load(ref_path)
    ref_data = ref_img.get_fdata()

    # Paths to the candidate direction files for this tract
    v1_path = os.path.join(direction_dir, f"{tract_name}_v1.nii.gz")
    v2_path = os.path.join(direction_dir, f"{tract_name}_v2.nii.gz")
    v3_path = os.path.join(direction_dir, f"{tract_name}_v3.nii.gz")

    if not (os.path.exists(v1_path) and os.path.exists(v2_path) and os.path.exists(v3_path)):
        print(f"Skipping {tract_name} since one of the candidate files is missing.")
        continue

    # Load v1, v2, v3 (each shape: X x Y x Z x 3)
    v1_data = nib.load(v1_path).get_fdata()
    v2_data = nib.load(v2_path).get_fdata()
    v3_data = nib.load(v3_path).get_fdata()

    # We will create an empty "dominant" array (same shape as ref_data)
    dominant_data = np.zeros_like(ref_data)

    # -------------------------------------------------
    # Identify nonzero voxels in the reference
    # -------------------------------------------------
    # norm of reference vectors
    ref_norm = np.linalg.norm(ref_data, axis=-1)

    # mask of nonzero voxels
    mask = ref_norm > 0

    # gather the (x, y, z) indices where the reference is nonzero
    nonzero_indices = np.argwhere(mask)  # shape: (N, 3), N = number of nonzero voxels

    # -------------------------------------------------
    # Iterate index-by-index (step-by-step)
    # -------------------------------------------------
    for idx in nonzero_indices:
        x, y, z = idx  # integer coordinates

        # Extract the reference 3D vector at this voxel
        ref_vec = ref_data[x, y, z, :]  # shape (3,)

        # Candidate vectors
        v1_vec = v1_data[x, y, z, :]  # shape (3,)
        v2_vec = v2_data[x, y, z, :]
        v3_vec = v3_data[x, y, z, :]

        # Compute norms
        ref_vec_norm = np.linalg.norm(ref_vec)
        v1_vec_norm  = np.linalg.norm(v1_vec)
        v2_vec_norm  = np.linalg.norm(v2_vec)
        v3_vec_norm  = np.linalg.norm(v3_vec)

        # Dot products
        dot1 = np.dot(ref_vec, v1_vec)
        dot2 = np.dot(ref_vec, v2_vec)
        dot3 = np.dot(ref_vec, v3_vec)

        # Avoid division by zero using a small epsilon or conditional checks
        # Here, we do a manual check to handle zero-norm candidates
        def safe_cosine(dot_val, norm1, norm2):
            if norm1 == 0 or norm2 == 0:
                return -np.inf  # so it won't be picked
            return dot_val / (norm1 * norm2)

        cos1 = safe_cosine(dot1, ref_vec_norm, v1_vec_norm)
        cos2 = safe_cosine(dot2, ref_vec_norm, v2_vec_norm)
        cos3 = safe_cosine(dot3, ref_vec_norm, v3_vec_norm)

        # Find the max among cos1, cos2, cos3
        cos_vals = [cos1, cos2, cos3]
        best_idx = np.argmax(cos_vals)  # 0, 1, or 2

        # Assign the chosen vector into dominant_data
        if best_idx == 0:
            dominant_data[x, y, z, :] = v1_vec
        elif best_idx == 1:
            dominant_data[x, y, z, :] = v2_vec
        else:
            dominant_data[x, y, z, :] = v3_vec

    # -------------------------------------------------
    # Save the output
    # -------------------------------------------------
    out_name = f"{tract_name}_dominant.nii.gz"
    out_path = os.path.join(output_dir, out_name)
    out_img  = nib.Nifti1Image(dominant_data, ref_img.affine)
    nib.save(out_img, out_path)
    print(f"Saved dominant output for {tract_name} to {out_path}")
