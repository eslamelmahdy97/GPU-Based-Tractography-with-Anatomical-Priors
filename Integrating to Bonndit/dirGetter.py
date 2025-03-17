from weakref import ref
import torch as T
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
from bonndit.models.LowRank import LowRankModel
from bonndit.models.TOM_Model import SimpleMLP
from bonndit.utils import tensor as bdt
mult = T.tensor(bdt.MULTIPLIER[4], device='cuda')**0.5
class DirectionGetter:
    def __init__(self, regularized_model, ref):
        pass
    
    def get_directions(self, fodfs):
        pass
    
class DirectionGetterLowRank(DirectionGetter):
    def __init__(self, model, data):
        self.model = LowRankModel([15, 1000, 1000, 1000, 9], 0.01)
        self.model.load_state_dict(T.load(model, weights_only=True), strict=False)


        self.model.eval()
        self.model.cuda()
        self.data = data 
    
    @T.no_grad()
    def get_directions(self, position):
        fodfs = T.as_tensor(self.data(position), dtype=T.float32, device='cuda')
        #l = T.norm(mult * fodfs, dim=-1)
        #fodfs /= l[:, None]
        output =  self.model(fodfs)
        #output *= l[:, None]
        output[T.norm(fodfs, dim=-1) == 0] = 0
        return output
    
    def _init_dir(self, ret_vals, seeds, where):
        """ Takes seeds, interpolates the fodfs and selects the largest direction as the initial direction, where
        "where" is True.

        Args:
            ret_vals (pytorch array): n x 3 array of directions
            seeds (pytorch array): n x 3 array of seeds in index space
            where (bools): n x 1 array of bools
        """
        ret = ret_vals[where.get()].to('cuda')
        current_fodfs = self.data(seeds[where.get(), :4])
        current_fodfs = T.as_tensor(current_fodfs, dtype=T.float32, device='cuda')
        #l = T.norm(mult * current_fodfs, dim=-1)
        #current_fodfs /= l[:, None]
        output = self.model(current_fodfs)
        #output *= l[:, None]
        nrm = T.zeros((output.shape[0], 3), dtype=T.float32).to('cuda')
        for i in range(3):
            nrm[:, i] = T.norm(output[:, i*3:(i+1)*3], dim=-1) 
        nrm = T.argmax(nrm, dim=-1)
        for i in range(3):
            ret[nrm ==i] = output[nrm == i, i*3:(i+1)*3]
        ret_vals[where.get()] = ret.to('cpu').detach()
        return ret_vals
    
    def init_dirs(self, seeds):
        ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32)
        where = cp.ones(seeds.shape[0], dtype=bool)
        return self._init_dir(ret_vals, seeds, where)
    
    ## TODO this has to be implemented!
  #  def nnls(self, position, dirs):
  #      """Takes the position and the directions and returns the nnls solution of the directions at the position.

  #      Args:
  #          position (pytorch array): n x 3 array of positions in index space
  #          dirs (pytorch array): n x 3 array of directions

  #     """
  #      fodfs = self.dataInterpolator(position)
  #      fodfs = T.tensor(fodfs.get(), dtype=T.float32).to('cuda')
  #      output = self.model(fodfs)
  #      output[T.norm(fodfs, dim=-1) == 0] = 0
  #      return T.nnls(output, dirs)
    
class DirectionGetterLowRankReg(DirectionGetterLowRank):
    def __init__(self, regularized_model, ref, model, data):
        super().__init__(model, data)
        self.regModel = LowRankModel([18, 1000, 1000, 1000, 9], 0.1)
        self.regModel.load_state_dict(T.load(regularized_model))
        self.regModel.cuda()
        self.regModel.eval()
        self.ref = ref
    
    @T.no_grad()
    def get_directions(self, position):
        ## If a reference direction is available, use it and use regularized model, else use the model
        fodfs = self.data(position)
        refs = self.ref(position)
        where = T.as_tensor(cp.linalg.norm(refs, axis=-1) < 0.1, device='cuda')
        refs = T.as_tensor(refs, dtype=T.float32, device='cuda')
        refs /= T.norm(refs, dim=-1)[:, None]
        refs = T.nan_to_num(refs)
        fodfs = T.as_tensor(fodfs, dtype=T.float32, device='cuda')
        fodfs_merged = T.hstack([fodfs[~where], refs[~where]])
        ## TODO Just works with rank 3
        output = T.zeros((fodfs.shape[0], 9), dtype=T.float32).to('cuda') 
        output[~where] =  self.regModel(fodfs_merged)
        output[where] = self.model(fodfs[where])
      #  output = self.model(fodfs)
        output[T.norm(fodfs, dim=-1) == 0] = 0
        return output
    
    def init_dirs(self, seeds):
        """Takes seeds, interopolates the reference directions, if no reference direction is available, uses 
        largest direction of the fodf as reference direction.

        Args:
            seeds (pytorch array): n x 3 array of seeds in index space
        """
        current_ref = self.ref(seeds[:, :4])
        where_ref = cp.linalg.norm(current_ref, axis=-1) > 0
        ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32)
        ret_vals[where_ref.get(), :] = T.tensor(current_ref[where_ref.get()].get(), dtype=T.float32)
        ret_vals[where_ref.get()] /= T.norm(ret_vals[where_ref.get()], dim=-1)[:, None]
        if (~where_ref).sum() != 0:
            ret_vals = self._init_dir(ret_vals, seeds, ~where_ref)
        return ret_vals
    ####################################################################################################
    

class TOMDirectionGetter(DirectionGetter):
    def __init__(self, model_path, fodf_data, tom_data, device='cuda'):
        """
        Args:
            model_path (str): Path to the pretrained model checkpoint (.pt file).
            fodf_data: An interpolator or callable that returns [15]-dimensional fODF features.
            tom_data: An interpolator or callable that returns [3]-dimensional TOM features.
            device (str): Device to run inference on (default 'cuda').
        """
        # Instantiate your pretrained model (SimpleMLP) with the correct architecture.
        self.model = SimpleMLP()  # Ensure this class matches your network architecture.
        # Load the checkpoint onto the specified device.
        self.model.load_state_dict(T.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        
        self.fodf_data = fodf_data  # Interpolator for fODF field (expects positions and returns [15] per position)
        self.tom_data = tom_data    # Interpolator for TOM field (expects positions and returns [3] per position)
        self.device = device

    @T.no_grad()
    def get_directions(self, position):
        """
        Args:
            position: Positions (e.g., a numpy array or cupy array) for which to interpolate features.
        Returns:
            A Torch tensor of shape [n, 9] with the model output (3 candidate directions per position).
        """
        # Interpolate the fODF features; expected output shape: [n, 15]
        fodf_features = T.as_tensor(self.fodf_data(position), dtype=T.float32, device=self.device)
        # Interpolate the TOM features; expected output shape: [n, 3]
        tom_features = T.as_tensor(self.tom_data(position), dtype=T.float32, device=self.device)
        # Concatenate along the feature dimension => shape [n, 18]
        input_features = T.hstack([fodf_features, tom_features])
        # Pass the concatenated input through the model => shape [n, 9]
        output = self.model(input_features)
        # For now, we return all 9 outputs (later you might select only the first 3 if desired)
        return output
    
    def _init_dir(self, ret_vals, seeds, where):
        """
        Helper function to initialize directions for seed points.
        
        This method does the following:
         - For seeds where initialization is needed (indicated by the boolean mask 'where'),
           it interpolates both the fODF and TOM features.
         - Concatenates these features to form 18-dimensional inputs.
         - Feeds the inputs through the model.
         - For each seed, reshapes the model output into 3 candidate directions (shape [3, 3]),
           computes the norm of each candidate, and selects the candidate with the highest norm.
         - Updates the ret_vals tensor with the chosen direction.
        
        Args:
            ret_vals (tensor): A tensor of shape [N, 3] initialized with zeros,
                               where N is the number of seeds.
            seeds (tensor or array): Seed positions.
            where (boolean mask): A mask (e.g., a Torch boolean tensor) indicating
                                  which seed indices to initialize.
        
        Returns:
            Updated ret_vals tensor with initial directions for the seeds.
        """
        # Convert selected seed directions to the device.
        ret = ret_vals[where].to(self.device)
        # Interpolate fODF features for the selected seeds.
        fodf_feats = T.as_tensor(self.fodf_data(seeds[where]), dtype=T.float32, device=self.device)
        # Interpolate TOM features for the selected seeds.
        tom_feats  = T.as_tensor(self.tom_data(seeds[where]), dtype=T.float32, device=self.device)
        # Concatenate the two sets to create 18-dimensional inputs.
        input_feats = T.hstack([fodf_feats, tom_feats])
        # Run the model to get outputs of shape [n, 9].
        output = self.model(input_feats)
        
        # For each seed, select one candidate direction.
        # Here we reshape each output vector (of length 9) into a matrix of shape [3, 3]
        # representing 3 candidate directions (each a 3D vector).
        directions = []
        for i in range(output.shape[0]):
            # Reshape the output of a single seed into 3 candidates.
            cand = output[i].view(3, 3)  # Shape becomes [3, 3].
            # Compute the norm of each candidate (resulting in a [3] vector).
            norms = T.norm(cand, dim=1)
            # Choose the candidate with the highest norm.
            best = cand[T.argmax(norms)]
            directions.append(best)
        # Stack all chosen directions into a tensor of shape [n, 3].
        directions = T.stack(directions)
        # Update the corresponding indices in ret_vals.
        ret = directions
        ret_vals[where] = ret.cpu().detach()
        return ret_vals

    def init_dirs(self, seeds):
        """
        Public method to initialize directions for all seed points.
        
        This function prepares a tensor of zeros for each seed and then calls
        the helper _init_dir to fill in the initial directions.
        
        Args:
            seeds: A tensor or array of seed positions.
        
        Returns:
            A tensor of shape [N, 3] containing the initial direction for each seed.
        """
        ret_vals = T.zeros((seeds.shape[0], 3), dtype=T.float32)
        # Here we assume that all seeds need to be initialized.
        # 'where' is a boolean mask of length N (all True).
        where = T.ones(seeds.shape[0], dtype=T.bool)
        return self._init_dir(ret_vals, seeds, where)
