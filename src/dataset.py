import os
import nrrd
import nibabel as nib
import torch
from torch.utils.data import Dataset

class fODFTOMDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir:
            Path to the main HCP_TractSeg directory which contains
            subfolders named numerically (e.g. '599469'). Each subject folder:
              - fodf.nrrd
              - A TOM folder containing .nii.gz volumes
        """
        self.root_dir = root_dir

        # CPU tensors
        self.fodf_data = {}   # fodf_data[subj_id] = shape [n_dirs, x, y, z]
        self.tom_volumes = {} # tom_volumes[(subj_id, tom_file)] = shape [x, y, z, n_dirs]

        # List of (subj_id, tom_file, x, y, z)
        self.samples = []

        # -----------------------------------------
        # Gather data across each subject directory
        # -----------------------------------------
        for subj_id in os.listdir(root_dir):
            subj_path = os.path.join(root_dir, subj_id)
            if not os.path.isdir(subj_path):
                continue  # not a folder

            # --- 1) Load FODF ---
            fodf_path = os.path.join(subj_path, 'fodf.nrrd')
            if not os.path.isfile(fodf_path):
                continue

            fodf_data_array, _ = nrrd.read(fodf_path)
            # Remove the first dimension 
            fodf_data_array = fodf_data_array[1:]  # shape [n_dirs, x, y, z]

            fodf_cpu_tensor = torch.tensor(
                fodf_data_array,
                dtype=torch.float32,
                device='cpu'
            )
            self.fodf_data[subj_id] = fodf_cpu_tensor

            # --- 2) Load TOM volumes ---
            tom_dir = os.path.join(subj_path, 'TOM')
            if not os.path.isdir(tom_dir):
                continue

            tom_files = [f for f in os.listdir(tom_dir) if f.endswith('.nii.gz')]
            for tom_file in tom_files:
                tom_path = os.path.join(tom_dir, tom_file)
                try:
                    tom_img = nib.load(tom_path)
                except nib.filebasedimages.ImageFileError as e:
                    print(f"Warning: Could not load '{tom_path}' ({e}). Skipping.")
                    continue

                tom_data = tom_img.get_fdata()
                tom_cpu_tensor = torch.tensor(
                    tom_data,
                    dtype=torch.float32,
                    device='cpu'
                )
                self.tom_volumes[(subj_id, tom_file)] = tom_cpu_tensor

                # --- 3) Identify non-zero voxels (on CPU) ---
                non_zero_mask = torch.any(tom_cpu_tensor != 0, dim=3)
                non_zero_coords = torch.nonzero(non_zero_mask)  # shape [N, 3]

                for coord in non_zero_coords:
                    x, y, z = coord.tolist()
                    self.samples.append((subj_id, tom_file, x, y, z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single voxel's data:
          - fODF (15 dims)
          - TOM last dimension (some # of channels, typically 3 for ref_dir)
        Concatenated into a single float tensor on CPU.
        """
        subj_id, tom_file, x, y, z = self.samples[idx]

        # fODF data: shape [n_dirs, x, y, z]
        voxel_fodf = self.fodf_data[subj_id][:, x, y, z]  # shape [15]

        # TOM data: shape [x, y, z, n_dirs]
        voxel_tom = self.tom_volumes[(subj_id, tom_file)][x, y, z, :]  # shape [n_tom_channels]

        # Concatenate on CPU
        input_data_cpu = torch.cat([voxel_fodf, voxel_tom], dim=0)

        return input_data_cpu