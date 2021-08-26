import scipy.interpolate as interp
import numpy as np
import torch
import PIL
from PIL import Image


class Lidar2D:
    """
    A 2D queryable lidar scanner module.
    """

    def __init__(
        self,
        img,
        num_beams,
        beam_length,
        beam_samps,
        samp_distribution_factor,
        collision_samps,
        fine_samps,
    ):
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
        self.beam_stop_thresh = 0.6

        self.img = img
        self.num_beams = num_beams
        self.beam_samps = beam_samps
        self.collision_samps = collision_samps
        self.fine_samps = fine_samps
        self.samp_df = samp_distribution_factor

        self.nx = self.img.shape[1]
        self.ny = self.img.shape[0]
        self.beam_len = beam_length

        self.xs = np.linspace(-1, 1, num=self.nx)
        self.ys = np.linspace(-1, 1, num=self.ny)

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
            t = np.linspace(0.0, 1.0, num=self.collision_samps).reshape(-1, 1)

            coarse_pnts = pos + t * np.repeat(
                beam_vec, self.collision_samps, axis=0
            )
            coarse_scan_vals = self.density.ev(
                coarse_pnts[:, 0], coarse_pnts[:, 1]
            ).reshape(-1, 1)
            coarse_hit_ind = np.argmax(
                coarse_scan_vals >= self.beam_stop_thresh
            )

            if coarse_hit_ind == 0:
                # No collision is detected, evenly sample across beam
                t = np.linspace(0.0, 1.0, self.beam_samps).reshape(-1, 1)
                pnts = pos + t * np.repeat(beam_vec, self.beam_samps, axis=0)
            else:
                # Collision detected by beam, fine sample to find a more accurate
                # distance to the collision.
                t = np.linspace(0.0, 1.0, self.fine_samps).reshape(-1, 1)
                coarse_coll_pnt = coarse_pnts[coarse_hit_ind, :].reshape(1, 2)
                last_empty = coarse_pnts[coarse_hit_ind - 1, :].reshape(1, 2)

                fine_pnts = last_empty + t * np.repeat(
                    (coarse_coll_pnt - last_empty),
                    self.fine_samps,
                    axis=0,
                )
                fine_scan_vals = self.density.ev(
                    fine_pnts[:, 0], fine_pnts[:, 1]
                ).reshape(-1, 1)
                fine_hit_ind = np.argmax(
                    fine_scan_vals >= self.beam_stop_thresh
                )

                collision_pnt = fine_pnts[fine_hit_ind, :].reshape(1, 2)

                # weighted sample between collision point and pos to
                # generate more data near walls.
                t_weighted = np.power(
                    np.linspace(0.0, 1.0, self.beam_samps), self.samp_df
                ).reshape(-1, 1)
                pnts = pos + t_weighted * np.repeat(
                    collision_pnt - pos, self.beam_samps, axis=0
                )

            scan_vals = self.density.ev(pnts[:, 0], pnts[:, 1]).reshape(-1, 1)
            beam_data.append(np.concatenate((pnts, scan_vals), axis=1))
        return np.vstack(beam_data)


class RandomPoseLidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        num_beams,
        beam_length,
        beam_samps,
        samp_distribution_factor,
        collision_samps,
        fine_samps,
        num_scans,
        round_density=True,
        border_width=0,
    ):
        super().__init__()
        self.img = np.asarray(Image.open(img_dir)).astype(float) / 255.0
        if border_width != 0:
            self.img[:, :border_width] = 1.0
            self.img[:border_width, :] = 1.0
            self.img[:, -border_width:-1] = 1.0
            self.img[-border_width:-1, :] = 1.0

        self.lidar = Lidar2D(
            self.img,
            num_beams,
            beam_length,
            beam_samps,
            samp_distribution_factor,
            collision_samps,
            fine_samps,
        )

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

        self.tds = torch.utils.data.TensorDataset(
            self.scans[:, :2], self.scans[:, 2]
        )

    def __getitem__(self, idx):
        return self.tds[idx]

    def __len__(self):
        return len(self.tds)


class TrajectoryLidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        num_beams,
        beam_length,
        beam_samps,
        samp_distribution_factor,
        collision_samps,
        fine_samps,
        waypoints,
        spline_res,
        round_density=True,
        border_width=0,
    ):
        super().__init__()
        self.img = np.asarray(Image.open(img_dir)).astype(float) / 255.0
        if border_width != 0:
            self.img[:, :border_width] = 1.0
            self.img[:border_width, :] = 1.0
            self.img[:, -border_width:-1] = 1.0
            self.img[-border_width:-1, :] = 1.0

        self.lidar = Lidar2D(
            self.img,
            num_beams,
            beam_length,
            beam_samps,
            samp_distribution_factor,
            collision_samps,
            fine_samps,
        )
        trajectory = interpolate_waypoints(
            waypoints[:, 0], waypoints[:, 1], spline_res
        )

        num_scans = trajectory.shape[0]

        self.scan_locs = trajectory
        scan_list = [
            self.lidar.scan(self.scan_locs[k, :].reshape(1, 2))
            for k in range(num_scans)
        ]

        self.scans = torch.from_numpy(np.vstack(scan_list))

        if round_density:
            self.scans[:, 2] = np.rint(self.scans[:, 2])

        self.tds = torch.utils.data.TensorDataset(
            self.scans[:, :2], self.scans[:, 2]
        )

    def __getitem__(self, idx):
        return self.tds[idx]

    def __len__(self):
        return len(self.tds)


def interpolate_waypoints(x, y, spline_res):
    i = np.arange(len(x))

    interp_i = np.linspace(0, i.max(), spline_res * i.max())

    xi = interp.interp1d(i, x, kind="cubic")(interp_i)
    yi = interp.interp1d(i, y, kind="cubic")(interp_i)

    return np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))
