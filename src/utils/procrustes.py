
import pytorch3d
import torch
from pytorch3d.ops import corresponding_points_alignment


def fit_transformations(
    source_pcds: torch.Tensor, # (B * P, N, 3)
    target_pcds: torch.Tensor, # (B * P, N, 3)
    part_lengths: torch.Tensor, # (B, P)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit rigid transformations from source to target point clouds per part.

    Args:
        source_pcds: Tensor of shape (B, N, 3) representing the source point clouds.
        target_pcds: Tensor of shape (B, N, 3) representing the target point clouds.
        part_lengths: Tensor of shape (B, P) giving the number of points in each part.
    Returns:
        rotations: Tensor of shape (B, P, 3, 3) representing the rotation matrices.
        translations: Tensor of shape (B, P, 3) representing the translation vectors.
    """
    # print("source_pcds", source_pcds.shape)
    # print("part_lengths", part_lengths.shape, part_lengths)
    BP, N, _ = source_pcds.shape
    # part_lengths -> mask padded points
    indices = torch.arange(N, device=source_pcds.device)
    weights = (indices[None, :] < part_lengths.reshape(-1, 1)).float()  # (BP, N)
    # print("weights", weights.shape)
    
    R, T, s = corresponding_points_alignment(
        X=source_pcds,
        Y=target_pcds,
        weights=weights,
        estimate_scale=True,
    )
    
    return R, T, s