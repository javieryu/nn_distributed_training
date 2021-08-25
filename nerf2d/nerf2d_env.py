import torch
import numpy as np

import torch_interpolations as ti


class NeRF2DEnv:
    """Takes a black and white image and interpolates.
    Using this class we can also keep track of the bounds of the
    scan environment etc."""

    def __init__(self, img, device, conf) -> None:
        # Load the image to device
        self.img = img.T.to(device)
        self.conf = conf

        # Setup the grid interpolator which will be the ground
        # truth density map.
        self.xlims = torch.tensor(conf["xlims"])
        self.ylims = torch.tensor(conf["ylims"])
        self.xs = torch.linspace(
            self.xlims[0], self.xlims[1], self.img.shape[0], device=device
        )
        self.ys = torch.linspace(
            self.ylims[0], self.ylims[1], self.img.shape[1], device=device
        )
        self.gi = ti.RegularGridInterpolator((self.xs, self.ys), self.img)

        # Create the lidar object
        self.lidar = Lidar(
            t_n=conf["t_near"], t_f=conf["t_far"], num_beams=conf["num_beams"]
        )

        # Mesh input
        # TODO: generate a meshgrid here for evaluation purposes

        X, Y = np.meshgrid(self.xs, self.ys)
        xlocs = X[::8, ::8].reshape(-1, 1)
        ylocs = Y[::8, ::8].reshape(-1, 1)
        mesh_poses = np.hstack((xlocs, ylocs))
        mesh_inputs = torch.Tensor(mesh_poses)
        self.mesh_inputs = mesh_inputs.to(device)

    def get_densities(self, pts, opacity=1.0):
        """Takes scan points in R2 and returns their
        densities.

        Args:
            pts (Tensor): Tensor with shape
                [batch, beam, num_points, 2 (x, y)]

        Returns: density as [batch, beams, num_points]
        """

        B, n_b, n_p, _ = pts.shape

        rho = torch.stack(
            [
                self.gi(
                    (
                        pts[ii, :, :, 0].reshape(-1),
                        pts[ii, :, :, 1].reshape(-1),
                    )
                ).reshape(n_b, n_p)
                for ii in range(B)
            ]
        )

        return rho.multiply_(opacity)


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
        self.bin_length = (t_f - t_n) / num_beams

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
        beam_angles = torch.zeros((B, 1), device=device) + torch.linspace(
            -np.pi, np.pi, self.num_beams
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
