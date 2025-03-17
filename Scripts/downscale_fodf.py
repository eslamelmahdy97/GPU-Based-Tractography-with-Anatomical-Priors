import nibabel as nib
import nrrd
import numpy as np

def downscale_fodf_entire(fodf_path, tom_path, output_path, scale_factor=0.5):
    

    # 1) Load the TOM (.nii.gz)
    tom_img = nib.load(tom_path)
    tom_data = tom_img.get_fdata()  # shape: [X, Y, Z] or [X, Y, Z, channels]

    # 2) Load the full fODF (.nrrd) -- expected shape: [16, X, Y, Z]
    fodf_data, fodf_header = nrrd.read(fodf_path)
    # For entire scaling, keep all 16 channels intact.
    expected_shape = fodf_data.shape  # expected: (16, X, Y, Z)

    # 3) Reorder the data to [X, Y, Z, 16] for voxelwise processing.
    fodf_data = np.moveaxis(fodf_data, 0, -1)

    # 4) Check that the spatial dimensions match between TOM and fODF
    if tom_data.ndim == 4:
        if fodf_data.shape[:3] != tom_data.shape[:3]:
            raise ValueError("Spatial dimensions do not match for 4D TOM.")
    elif tom_data.ndim == 3:
        if fodf_data.shape[:3] != tom_data.shape:
            raise ValueError("Spatial dimensions do not match for 3D TOM.")
    else:
        raise ValueError("Unexpected number of dimensions in TOM data.")

    # 5) Build non-zero masks:
    
    if tom_data.ndim == 4:
        tom_nonzero_mask = np.any(tom_data != 0, axis=3)
    else:
        tom_nonzero_mask = (tom_data != 0)
    
    # For fODF (now shape [X, Y, Z, 16]): mark voxel active if any channel is nonzero.
    fodf_nonzero_mask = np.any(fodf_data != 0, axis=3)
    
    # 6) Create a common mask: voxels that are active in both TOM and fODF.
    common_mask = np.logical_and(tom_nonzero_mask, fodf_nonzero_mask)

    # 7) Downscale all channels in the common voxels by the scale factor.
    fodf_data[common_mask, :] *= scale_factor

    # 8) Reorder the data back to the original ordering: [16, X, Y, Z]
    final_fodf = np.moveaxis(fodf_data, -1, 0)

    # 9) Verify that the final shape matches the expected shape.
    if final_fodf.shape != expected_shape:
        raise ValueError(f"Processed fODF shape {final_fodf.shape} does not match expected shape {expected_shape}.")

    # 10) Write out the downscaled fODF file.
    nrrd.write(output_path, final_fodf, fodf_header)
    print(f"Downscaled fODF written to {output_path}, final shape: {final_fodf.shape}")

if __name__ == "__main__":
    # Example usage:
    fodf_input = "odf.nrrd"          # input fODF with shape [16, X, Y, Z]
    tom_input = "/home/s65eelma/TOM/CC_1.nii.gz"
    fodf_output = "0_adjusted_fODF.nrrd"

    
    scaling_factor = 0

    downscale_fodf_entire(
        fodf_path=fodf_input,
        tom_path=tom_input,
        output_path=fodf_output,
        scale_factor=scaling_factor
    )
