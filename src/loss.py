import torch

def loss_fn(input_data, output, nu):
    """
    input_data: shape [batch, 18], first 15 are fODF, last 3 are ref_dir
    output: shape [batch, 9], split into v1, v2, v3 (each 3D)
    nu: penalty weight
    """
    fODF = input_data[:, :15]
    ref_dir = input_data[:, 15:]   # shape [batch, 3]

    v1 = output[:, 0:3]
    v2 = output[:, 3:6]
    v3 = output[:, 6:9]

    def monomials(vec):
        x = vec[:, 0]
        y = vec[:, 1]
        z = vec[:, 2]
        return torch.stack([
            x**4,
            x**3 * y,
            x**3 * z,
            x**2 * y**2,
            x**2 * y * z,
            x**2 * z**2,
            x * y**3,
            x * y**2 * z,
            x * y * z**2,
            x * z**3,
            y**4,
            y**3 * z,
            y**2 * z**2,
            y * z**3,
            z**4
        ], dim=1)
    # For each of the three output vectors (v1, v2, v3), compute their fourth-order monomials.
    v1_m = monomials(v1)
    v2_m = monomials(v2)
    v3_m = monomials(v3)

    total_monomials = v1_m + v2_m + v3_m
    diff = fODF - total_monomials

    # Weighted MSE for order-4 terms
    order_4_mult = torch.tensor([
        1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0,
        4.0, 1.0, 4.0, 6.0, 4.0, 1.0
    ], dtype=torch.float32, device=diff.device).view(1, 15)

    weighted_diff = order_4_mult * diff**2
    reconstruction_loss = weighted_diff.mean()

    # Direction penalty for v1 vs. ref_dir
    v1_norm = torch.norm(v1, dim=1, keepdim=True) + 1e-8
    ref_norm = torch.norm(ref_dir, dim=1, keepdim=True) + 1e-8
    cos_theta = torch.sum(v1 * ref_dir, dim=1) / (v1_norm.squeeze(-1)*ref_norm.squeeze(-1))

    penalty = nu * (1 - cos_theta**2)
    regularization_term = penalty.mean()

    total_loss = reconstruction_loss + regularization_term
    return total_loss