import numpy as np
import torch


class Lidar:
    """
    Implements 2D lidar for MNIST images.
    """

    def __init__(self, t_n=0.1, t_f=1.0, num_beams=10, fov=np.pi / 2):
        self.t_n = t_n
        self.t_f = t_f
        self.num_beams = num_beams
        self.fov = fov

    def generate_scan_points(self, pos, num_points=25):
        """
        Generates sampling points from a given lidar position
        and number of points per beam.

        Returns both the point positions in R^2, and their distances
        from the lidar.

        Args:
            pos: batch of lidar positions, a torch Tensor with shape [B, 2].
            num_points: points to sample per ray.
        """
        B, _ = pos.shape
        device, dtype = pos.device, pos.dtype

        # compute view angle looking at origin
        start_angles = np.pi + torch.atan2(pos[:, 1], pos[:, 0]).unsqueeze(-1)

        # equally space beams over angle of self.fov
        beam_angles = start_angles + torch.linspace(
            -self.fov / 2, self.fov / 2, self.num_beams
        ).reshape(1, -1).to(device, dtype)

        bin_length = (self.t_f - self.t_n) / num_points

        # compute beam rays
        beams = torch.stack(
            (torch.cos(beam_angles), torch.sin(beam_angles)), dim=-1
        ).reshape(B, self.num_beams, 1, 2)

        dists = (
            torch.linspace(self.t_n, self.t_f, num_points)
            .reshape(1, 1, num_points)
            .to(device, dtype)
        )

        dists = dists + bin_length * torch.rand(
            B, self.num_beams, num_points
        ).to(device, dtype)

        return (
            pos.reshape(B, 1, 1, 2)
            + dists.reshape(B, -1, num_points, 1) * beams,
            beams,
            dists,
        )
