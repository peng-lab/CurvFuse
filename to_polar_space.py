import torch
import torch.nn.functional as F


def cartesian_to_polar(x, T=360, R=None, center=None, device="cuda"):
    """
    Convert 2D image (or feature map) from Cartesian to polar coordinates.

    Args:
        x: torch.Tensor, [H,W] or [1,1,H,W]
        T: number of angular samples (theta)
        R: number of radial samples (if None, use min(H,W)//2)
        center: (cx,cy) in pixel coordinates; if None, image center is used
    Returns:
        x_polar: torch.Tensor, [T,R]  (if input 2D) or [1,1,T,R]
    """

    x = torch.from_numpy(x).to(device)
    # ensure shape [1,1,H,W]
    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        x = x[:, None, ...]
    B, C, H, W = x.shape
    device = x.device

    # radius and angle sampling
    if R is None:
        R = int(min(H, W) / 2)
    thetas = torch.linspace(-torch.pi, torch.pi, T, device=device)
    radii = torch.linspace(0, 1, R, device=device)
    Rmax = min(H, W) / 2.0

    # define center
    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    # build sampling grid (theta,r)
    rr = radii * Rmax
    cos_t, sin_t = torch.cos(thetas), torch.sin(thetas)
    X = rr[None, :].repeat(T, 1) * cos_t[:, None] + cx
    Y = rr[None, :].repeat(T, 1) * sin_t[:, None] + cy

    # normalize to [-1,1]
    grid_x = 2 * (X / (W - 1)) - 1
    grid_y = 2 * (Y / (H - 1)) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [T,R,2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,T,R,2]

    # sample
    x_polar = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [B,C,T,R]

    return x_polar if C > 1 or B > 1 else x_polar[0, 0]
