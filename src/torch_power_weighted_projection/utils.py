"""Utility functions for power weighted projection."""

import torch


def zyz_to_rotation_matrix(alpha, beta, gamma, device='cuda'):
    """Convert ZYZ Euler angles to rotation matrix.

    Args:
        alpha: First rotation angle around Z-axis (in radians)
        beta: Second rotation angle around Y-axis (in radians)
        gamma: Third rotation angle around Z-axis (in radians)
        device: Device to use for computation ('cuda' or 'cpu')

    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
    ca, sa = torch.cos(torch.tensor(alpha, device=device)), torch.sin(torch.tensor(alpha, device=device))
    cb, sb = torch.cos(torch.tensor(beta, device=device)), torch.sin(torch.tensor(beta, device=device))
    cg, sg = torch.cos(torch.tensor(gamma, device=device)), torch.sin(torch.tensor(gamma, device=device))

    Rz_alpha = torch.tensor([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], device=device)
    Ry_beta = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], device=device)
    Rz_gamma = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], device=device)

    return Rz_alpha @ Ry_beta @ Rz_gamma
