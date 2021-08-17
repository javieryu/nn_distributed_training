import numpy as np
import torch


class Lidar:
    """
    Implements 2D lidar for MNIST images.

    Adapted with permission by Javier Yu, from code
    written originally by Preston Culbertson.
    """

    def __init__(self, t_n=0.1, t_f=1.0, num_beams=10):
        self.t_n = t_n
        self.t_f = t_f
        self.num_beams = num_beams

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

        # equally space beams over angle of self.fov
        beam_angles = (
            torch.linspace(-np.pi, np.pi, self.num_beams)
            .reshape(1, -1)
            .to(device, dtype)
        )

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
