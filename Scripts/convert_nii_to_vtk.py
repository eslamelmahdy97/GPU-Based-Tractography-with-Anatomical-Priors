import os
import nibabel as nib
import numpy as np
import pyvista as pv

def convert_nii_to_vtk(input_dir, output_dir="dom_vtk"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all .nii.gz files in the input directory
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    if not nii_files:
        print("No .nii.gz files found in the directory.")
        return

    for nii_file in nii_files:
        input_path = os.path.join(input_dir, nii_file)
        output_path = os.path.join(output_dir, nii_file.replace('.nii.gz', '.vtk'))

        print(f"Processing {nii_file}...")

        # Load NIfTI data
        img = nib.load(input_path)
        data = img.get_fdata()

        # Check shape [X, Y, Z, 3]
        if data.ndim != 4 or data.shape[3] != 3:
            print(f"Skipping {nii_file}, expected shape [X, Y, Z, 3] but got {data.shape}")
            continue

        x_dim, y_dim, z_dim, _ = data.shape

        # Reshape directional vectors into (num_points, 3)
        vectors = data.reshape(-1, 3)

        # Create a structured grid
        grid = pv.StructuredGrid()

        # Build the (i, j, k) indices for every voxel
        i, j, k = np.meshgrid(
            np.arange(x_dim),
            np.arange(y_dim),
            np.arange(z_dim),
            indexing='ij'
        )

        # Apply the affine transformation matrix from the NIfTI file to convert voxel coordinates to real-world coordinates
        affine = img.affine
        ones = np.ones((i.size, 1), dtype=np.float32)
        ijk_hom = np.column_stack((i.ravel(), j.ravel(), k.ravel(), ones.ravel()))
        xyz = (affine @ ijk_hom.T).T[:, :3]  # shape: (num_points, 3)

        # Assign points and dimensions to the structured grid
        grid.points = xyz
        grid.dimensions = [x_dim, y_dim, z_dim]

        # Attach the vector data as point data
        grid.point_data["fiber_directions"] = vectors

        # Save the grid to a .vtk file for use in ParaView
        grid.save(output_path)
        print(f"Saved {output_path}")

    print("Conversion completed.")

if __name__ == "__main__":
    input_directory = "/home/s65eelma/Dominant_Directions" 
    convert_nii_to_vtk(input_directory)
