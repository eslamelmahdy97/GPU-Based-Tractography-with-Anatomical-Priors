import os
import torch
import torch.optim as optim
import nibabel as nib
import numpy as np
import nrrd

from model import SimpleMLP
from utils import load_checkpoint

def save_vector_field_as_nii(vec_array_4d, filename, reference_img=None):
    """
    vec_array_4d: shape [X, Y, Z, 3], np.float32
      We interpret the last dim (3) as vector components.

    filename: The .nii.gz path to write
    reference_img (optional): nibabel object to copy affine/header from.
      If None, we'll use an identity affine.
    """
    X, Y, Z, C = vec_array_4d.shape
    assert C == 3, f"Expected last dimension=3, got {C}"

    # create a Nifti1Image. By default, use identity affine if no reference provided.
    if reference_img is not None:
        affine = reference_img.affine
    else:
        affine = np.eye(4, dtype=np.float32)

    nifti_img = nib.Nifti1Image(vec_array_4d, affine)
    nib.save(nifti_img, filename)
    print(f"Saved {filename} with shape {X}x{Y}x{Z}x{C} (NIfTI).")
    
    
def run_inference_for_subject(
    fodf_path,       # .nrrd file for this subject's fodf, shape [16, X, Y, Z] but we'll drop the first dim => [15, X, Y, Z]
    tracts_dir,      # directory with multiple .nii.gz, each is a separate tract with shape [X, Y, Z, 3]
    checkpoint_path, # path to 'checkpoint.pt'
    output_dir,      # where to save .nii.gz files
    device='cuda'
):
    """
    Loads fodf, loads each tract .nii.gz, does inference with your MLP, 
    saves predicted (v1, v2, v3) as .nii.gz volumes of shape [X, Y, Z, 3].
    """

    # 0) Create output_dir if needed
    os.makedirs(output_dir, exist_ok=True)

    ##########################################################################
    # 1) Load the fodf onto GPU
    ##########################################################################
    print(f"[INFO] Loading fodf from {fodf_path} onto GPU ...")
    fodf_data_array, _ = nrrd.read(fodf_path)
    print(f"fodf shape:{fodf_data_array.shape}")
    fodf_data_array = fodf_data_array[1:]       # shape [15, X, Y, Z]
    fodf_tensor_gpu = torch.tensor(fodf_data_array, dtype=torch.float32, device=device)

    X = fodf_tensor_gpu.shape[1]
    Y = fodf_tensor_gpu.shape[2]
    Z = fodf_tensor_gpu.shape[3]
    print(f"[INFO] fodf shape => [15, {X}, {Y}, {Z}] on {device}")

    ##########################################################################
    # 2) Load the Model + Checkpoint
    ##########################################################################
    print(f"[INFO] Loading model checkpoint {checkpoint_path}")
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    load_checkpoint(model, optimizer, checkpoint_path, device=device)
    model.eval()

    ##########################################################################
    # 3) For each tract in tracts_dir, run inference
    ##########################################################################
    tract_files = [f for f in os.listdir(tracts_dir) if f.endswith('.nii.gz')]
    if not tract_files:
        print(f"[WARNING] No .nii.gz found in {tracts_dir}")
        return

    for tract_filename in tract_files:
        tract_path = os.path.join(tracts_dir, tract_filename)
        base_name = os.path.splitext(os.path.splitext(tract_filename)[0])[0]  # remove .nii.gz
        print(f"\n[INFO] Processing tract file: {tract_path}")

        # 3a) Load tract onto GPU
        try:
            tract_img = nib.load(tract_path)
        except nib.filebasedimages.ImageFileError as e:
            print(f"Warning: Could not load '{tract_path}' ({e}). Skipping.")
            continue
        tract_data = tract_img.get_fdata()   # shape [X, Y, Z, 3]
        # Move entire volume to GPU
        tract_tensor_gpu = torch.tensor(tract_data, dtype=torch.float32, device=device)
        if tract_tensor_gpu.shape != (X, Y, Z, 3):
            print(f"[WARNING] shape mismatch, got {tract_tensor_gpu.shape}, "
                  f"expected [X, Y, Z, 3] = [{X}, {Y}, {Z}, 3]")
        
        # 3b) Identify non-zero coords
        #    We'll consider a voxel "active" if any channel != 0
        #    This yields shape [X, Y, Z], True where non-zero
        non_zero_mask = torch.any(tract_tensor_gpu != 0, dim=3)
        non_zero_coords = torch.nonzero(non_zero_mask)  # shape [N, 3]

        # Prepare outputs (on GPU) => shape [X, Y, Z, 3]
        pred_v1_gpu = torch.zeros((X, Y, Z, 3), dtype=torch.float32, device=device)
        pred_v2_gpu = torch.zeros((X, Y, Z, 3), dtype=torch.float32, device=device)
        pred_v3_gpu = torch.zeros((X, Y, Z, 3), dtype=torch.float32, device=device)

        # 3c) Inference over each voxel
        with torch.no_grad():
            for coord in non_zero_coords:
                x, y, z = coord.tolist()
                # gather 15 from fodf
                voxel_fodf = fodf_tensor_gpu[:, x, y, z]  # shape [15]
                # gather 3 from tract
                voxel_tom = tract_tensor_gpu[x, y, z, :]  # shape [3]
                # concat => shape [18]
                input_18 = torch.cat([voxel_fodf, voxel_tom], dim=0).unsqueeze(0)  # shape [1, 18]
                output_9 = model(input_18)  # shape [1, 9]

                v1 = output_9[0, 0:3]
                v2 = output_9[0, 3:6]
                v3 = output_9[0, 6:9]

                pred_v1_gpu[x, y, z, :] = v1
                pred_v2_gpu[x, y, z, :] = v2
                pred_v3_gpu[x, y, z, :] = v3

        # 3d) Convert GPU results back to CPU, then to NumPy for saving
        pred_v1_cpu = pred_v1_gpu.cpu().numpy()  # shape [X, Y, Z, 3]
        pred_v2_cpu = pred_v2_gpu.cpu().numpy()
        pred_v3_cpu = pred_v3_gpu.cpu().numpy()

        # 3e) Save as NIfTI
        out_v1_nii = os.path.join(output_dir, f"{base_name}_v1.nii.gz")
        out_v2_nii = os.path.join(output_dir, f"{base_name}_v2.nii.gz")
        out_v3_nii = os.path.join(output_dir, f"{base_name}_v3.nii.gz")

        # Use the same reference affine from tract_img if you want consistent orientation
        save_vector_field_as_nii(pred_v1_cpu, out_v1_nii, reference_img=tract_img)
        save_vector_field_as_nii(pred_v2_cpu, out_v2_nii, reference_img=tract_img)
        save_vector_field_as_nii(pred_v3_cpu, out_v3_nii, reference_img=tract_img)

        print(f"[INFO] Done: {tract_filename} => {out_v1_nii}, {out_v2_nii}, {out_v3_nii}")

if __name__ == "__main__":
  
    fodf_path='/home/s65eelma/odf.nrrd'      
    tracts_dir= '/home/s65eelma/TOM'    
    checkpoint_path= '/home/s65eelma/OLD/checkpoint.pt'
    output_dir= '/home/s65eelma/OLD_output'
    device = 'cuda'
    print(f"Using device: {device}")

    run_inference_for_subject(
        fodf_path,
        tracts_dir,
        checkpoint_path,
        output_dir,
        device=device
    )
    print("[INFO] Inference completed.")
