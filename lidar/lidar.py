import scipy.interpolate as interp
import numpy as np


class Lidar2D:
    """
    A 2D queryable lidar scanner module.
    """

    def __init__(self, img, num_beams, scan_dist_scale, beam_samps):
        """
        Args:
            img (np.array): Array with size (nx, ny),
            and values in the range [0.0, 1.0]

            num_beams (int): Number of beams in a single lidar scan.
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
        if self.density.ev(pos[0, 0], pos[0, 1]) >= self.beam_stop_thresh:
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
