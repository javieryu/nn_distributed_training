import scipy.interpolate as interp
import numpy as np
import torch
import PIL
from PIL import Image


class Lidar2D:
    """
    A 2D queryable lidar scanner module.
    """

    def __init__(self, img, num_beams, scan_dist_scale, beam_samps):
        """Setting up the lidar scanner

        Args:
            img (np.array): Array of densities (between 0 and 1)
             in image coordinates
            num_beams (int): set the number of beams in a single scan.
            scan_dist_scale (float): length of a single beam as a
             percentage of the length of the largest dimension of the image.
            beam_samps (int): Number of samples along each beam. Tuning this parameter
             will effect whether thin walls are detected.
        """
        self.beam_stop_thresh = 0.5

        self.img = img
        self.num_beams = num_beams
        self.beam_samps = beam_samps

        self.nx = self.img.shape[1]
        self.ny = self.img.shape[0]
        self.beam_len = scan_dist_scale * max(self.nx, self.ny)

        self.ys = self.ny * np.linspace(-0.5, 0.5, num=self.ny)
        self.xs = self.nx * np.linspace(-0.5, 0.5, num=self.nx)

        self.density = interp.RectBivariateSpline(self.xs, self.ys, self.img.T)

    def scan(self, pos):
        """Scans from a given coordinate

        Args:
            pos (np.array): an array with dims (1, 2) indicating the (x, y)
             position of the scan.

        Raises:
            NameError: Errors if a scan is called from inside a wall or outside
            of the image domain.

        Returns:
            (np.array): an array with dims (z, 3) where z is the number of scanned
             points that may vary between scans because of beams ending early when
             they hit walls.
        """
        if self.density.ev(pos[0, 0], pos[0, 1]) >= self.beam_stop_thresh:
            print(pos)
            raise NameError("Cannot lidar scan from point with high density.")
        angs = np.linspace(-np.pi, np.pi, num=self.num_beams, endpoint=False)

        beam_data = []
        for i in range(self.num_beams):
            beam_vec = self.beam_len * np.array(
                [np.cos(angs[i]), np.sin(angs[i])]
            ).reshape(1, -1)
            t = np.linspace(0.0, 1.0, num=self.beam_samps).reshape(-1, 1)

            pnts = pos + t * np.repeat(beam_vec, self.beam_samps, axis=0)
            scan_vals = self.density.ev(pnts[:, 0], pnts[:, 1]).reshape(-1, 1)
            hit_ind = np.argmax(scan_vals >= self.beam_stop_thresh)

            if hit_ind == 0:
                slice_ind = scan_vals.shape[0]
            else:
                slice_ind = hit_ind + 1

            beam_data.append(
                np.concatenate(
                    (pnts[:slice_ind, :], scan_vals[:slice_ind]), axis=1
                )
            )

        return np.vstack(beam_data)


class RandomPoseLidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        num_beams,
        scan_dist_scale,
        beam_samps,
        num_scans,
        round_density=True,
    ):
        super().__init__()
        self.img = np.asarray(Image.open(img_dir)).astype(float) / 255.0
        self.lidar = Lidar2D(self.img, num_beams, scan_dist_scale, beam_samps)

        # Generate Scan Coordinates
        c = 0
        locs = []
        while c < num_scans:
            xsamps = np.random.choice(self.lidar.xs, num_scans)
            ysamps = np.random.choice(self.lidar.ys, num_scans)
            psamps = self.lidar.density.ev(xsamps, ysamps)
            mask = psamps < 0.5
            c += np.sum(mask)

            locs.append(
                np.hstack(
                    [xsamps[mask].reshape(-1, 1), ysamps[mask].reshape(-1, 1)]
                )
            )

        self.scan_locs = np.vstack(locs)[:num_scans, :]

        scan_list = []
        for k in range(num_scans):
            pos = self.scan_locs[k, :].reshape(1, 2)
            scan_list.append(self.lidar.scan(pos))

        self.scans = torch.from_numpy(np.vstack(scan_list))

        if round_density:
            self.scans[:, 2] = np.rint(self.scans[:, 2])

    def __getitem__(self, idx):
        meta_dict = {
            "density": self.scans[idx, 2].reshape(1, -1),
            "position": self.scans[idx, :2].reshape(1, -1),
        }
        return meta_dict

    def __len__(self):
        return self.scans.shape[0]


class TrajectoryLidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        num_beams,
        scan_dist_scale,
        beam_samps,
        trajectory,
        round_density=True,
    ):
        super().__init__()
        self.img = np.asarray(Image.open(img_dir)).astype(float) / 255.0
        self.lidar = Lidar2D(self.img, num_beams, scan_dist_scale, beam_samps)
        self.traj = trajectory

        self.traj = np.flip(self.traj)
        self.traj[:, 0] -= self.lidar.nx

        num_scans = trajectory.shape[0]
        scan_list = []
        for k in range(num_scans):
            pos = trajectory[k, :].reshape(1, 2)
            scan_list.append(self.lidar.scan(pos))

        self.scans = torch.from_numpy(np.vstack(scan_list))

        if round_density:
            self.scans[:, 2] = np.rint(self.scans[:, 2])

    def __getitem__(self, idx):
        meta_dict = {
            "density": self.scans[idx, 2].reshape(1, -1),
            "position": self.scans[idx, :2].reshape(1, -1),
        }
        return meta_dict

    def __len__(self):
        return self.scans.shape[0]