import numpy as np
import torch

# TODO(pculbert): clean up these imports


def pos_to_coord(x, im_w=2.0, im_h=2.0, img_size=28, scale=1.0):
    """
    Casts position x in [-im_w/2, im_w/2] x [-im_h/2, im_h/2]
    to its coordinate in the image frame (bottom left is (0,0),
    top is (img_size, img_size)).
    """
    x_q1 = scale * (x + np.array([im_w, im_h]) / 2) / np.array([im_w, im_h])
    return x_q1 * img_size


def region_pixels(center, w, h, n_x, n_y):
    """
    Generates the center of each pixel in the "world" frame.
    """
    p_w = w / n_x
    p_h = h / n_y

    pxs = np.linspace(
        center[0] + (p_w - w) / 2, center[0] + (w - p_w) / 2, num=n_x
    )
    pys = np.linspace(
        center[1] + (p_h - h) / 2, center[1] + (h - p_h) / 2, num=n_y
    )

    X, Y = np.meshgrid(pxs, pys)

    return np.stack((X.flatten(), Y.flatten()))


def get_image_densities(points, images, opacity=1e2):
    """
    Maps points to their occupancy values in a stack of images.

    Args:
        points: torch Tensor of query points, in [-1, 1] x [-1, 1],
            shape [B, n_r, n_p, 2].
        images: torch Tensor of images, shape [B, H, W]. assume images are
            scaled to lie in [-1, 1] x [-1, 1]
        opacity: factor to scale image intensities by; lower values allow
            beams to pass through object more easily.

    Returns the occupancy value of each point's corresponding pixel in the
    batch of images.
    """
    batch_size, H, W = images.shape
    _, n_r, n_p, _ = points.shape

    dx = 2 / W
    dy = 2 / H

    c_x = (torch.floor((points[:, :, :, 0] + 1) / dx) - 1).clamp(0, W - 1)
    c_y = (torch.floor((points[:, :, :, 1] + 1) / dy) - 1).clamp(0, H - 1)

    c_p = (W * (H - 1 - c_y) + c_x).long()

    occs = torch.gather(
        images.reshape(batch_size, 1, -1).expand(batch_size, n_r, -1), 2, c_p
    )

    return opacity * occs


def depth_from_densities(dists, densities):
    """
    Computes mean and variance of depth along rays using sampled
    distances and their corresponding volume densities.

    Args:
        dists: sampled distances along each ray; torch Tensor of shape
               (B, n_r, n_p), where B is batch size, n_r is the number
               of rays, and n_p is the number of points sampled along
               each ray.
        densities: volumetric density, measured at each sampled point, also
                shape (B, n_r, n_p)
    """
    device = dists.device
    deltas = dists[:, :, 1:] - dists[:, :, :-1]
    occs = 1 - torch.exp(-densities[:, :, :-1] * deltas)
    inv_occs = torch.cumprod(1 - occs, -1)
    # inv_occs[:, :, 0] = 1.0
    inv_occs = torch.cat(
        (
            torch.ones_like(inv_occs[..., -1], device=device).unsqueeze(-1),
            inv_occs[..., :-1],
        ),
        dim=-1,
    )
    weights = occs * inv_occs

    weights_norm = weights / (1e-8 + torch.sum(weights, dim=-1, keepdim=True))

    depth_mean = torch.sum(dists[:, :, :-1] * weights_norm, dim=-1)
    depth_var = torch.sum(
        weights_norm
        * torch.square(dists[:, :, :-1] - depth_mean.unsqueeze(-1)),
        dim=-1,
    )

    return depth_mean, depth_var, weights


def fine_sample(weights, num_samples=25):
    """
    Samples (via rejection) from new_points bins according to their weight.
    """
    inds = torch.stack(
        [
            torch.multinomial(
                1e-8 + weights[ii, :, :], num_samples, replacement=True
            )
            for ii in range(weights.shape[0])
        ]
    )
    return inds


def neural_render(model, pos, lidar, num_coarse, num_fine, importance=False):
    coarse_pts, beams, coarse_dists = lidar.generate_scan_points(
        pos, num_points=num_coarse
    )
    B, n_b, _, _ = coarse_pts.shape
    device = next(model.parameters()).device

    if importance:
        coarse_rho = model.forward(coarse_pts).squeeze()

        with torch.no_grad():
            _, _, coarse_weights = depth_from_densities(
                coarse_dists, coarse_rho
            )
            fine_inds = fine_sample(coarse_weights, num_samples=num_fine)
            fine_dists = torch.gather(
                coarse_dists, -1, fine_inds
            ) + lidar.bin_length * torch.rand(fine_inds.shape, device=device)
            fine_points = (
                pos.reshape(B, 1, 1, 2)
                + fine_dists.reshape(B, n_b, -1, 1) * beams
            )

        fine_rho = model.forward(fine_points).squeeze()

        with torch.no_grad():
            all_dists = torch.cat((coarse_dists, fine_dists), axis=-1)
            all_dists, sinds = all_dists.sort()

        all_rho = torch.cat((coarse_rho, fine_rho), axis=-1)
        all_rho = torch.gather(all_rho, -1, sinds)
    else:
        all_dists = coarse_dists
        all_rho = model.forward(coarse_pts).squeeze()

    d_mean, d_var, _ = depth_from_densities(all_dists, all_rho)
    return d_mean, d_var
