#!/usr/bin/env python3

import sys
import numpy as np
import nibabel as nib
import nrrd

def main():
    # Paths to input/output files
    wm_nrrd_path = "/home/s65eelma/HCP_TractSeg/599469/wmvolume.nrrd"     # 3D WM volume
    tom_nii_path = "/home/s65eelma/HCP_TractSeg/599469/TOM/AF_left.nii.gz"       # 4D TOM volume
    out_nrrd_path = "599wmvolume_maskedAF.nrrd"
    
    # Read WM volume from NRRD
    print(f"Reading WM volume from {wm_nrrd_path} ...")
    wm_data, wm_header = nrrd.read(wm_nrrd_path)
    print(f"wm_data shape = {wm_data.shape} (expected 3D)")

    # Read TOM volume from NIfTI (4D: x, y, z, 3)
    print(f"Reading TOM volume from {tom_nii_path} ...")
    tom_img = nib.load(tom_nii_path)
    tom_data = tom_img.get_fdata()  # shape: (x, y, z, n_dirs)
    print(f"tom_data shape = {tom_data.shape} (expected 4D, last dim = 3)")

    # Basic shape check (3D portion only)
    if tom_data.shape[:3] != wm_data.shape:
        print("ERROR: The first three dimensions of TOM do not match the WM volume.")
        print(f"WM shape: {wm_data.shape}")
        print(f"TOM shape (3D portion): {tom_data.shape[:3]}")
        sys.exit(1)

    # 1) Compute the vector norm across the last dimension (the 3 directional components)
    #    This results in a 3D array.
    print("Computing vector norm of the TOM across its last dimension...")
    tom_norm = np.sqrt(np.sum(tom_data**2, axis=3))  # shape: (145, 174, 145)

    # 2) Create a 3D mask based on whether the norm is > 0
    
    print("Creating a mask from TOM norm...")
    mask_3d = (tom_norm > 0).astype(wm_data.dtype)  # shape: (145, 174, 145)

    # 3) Multiply the WM data by the mask
    print("Applying the mask to the WM volume...")
    masked_wm_data = wm_data * mask_3d

    # 4) Write the masked volume as NRRD
    print(f"Saving masked WM volume to {out_nrrd_path} ...")
    nrrd.write(out_nrrd_path, masked_wm_data, header=wm_header)

    print("Done.")

if __name__ == "__main__":
    main()
