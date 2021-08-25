import torch
import nerf2d_utils


class NeRF2DDataset(torch.utils.data.Dataset):
    """When generating NeRF data this allows us
        to pre-compute the ground truth data for scans
    from a list of poses in a 2D image.

        Args:
            torch ([type]): [description]
    """

    def __init__(self, nerf2denv, device, conf, trajectory=None) -> None:
        super().__init__()

        self.nenv = nerf2denv
        self.conf = conf
        self.device = device

        if conf["scan_loc_type"] == "random":
            c = 0
            num_scans = conf["num_scans"]

            locs = []
            xd = self.nenv.xlims[1] - self.nenv.xlims[0]
            yd = self.nenv.ylims[1] - self.nenv.ylims[0]
            x0, y0 = self.nenv.xlims[0], self.nenv.ylims[0]
            while c < num_scans:
                xsamps = xd * torch.rand(num_scans, device=device) + x0
                ysamps = yd * torch.rand(num_scans, device=device) + y0

                psamps = self.nenv.gi((xsamps, ysamps))
                mask = psamps < 0.5
                c += torch.sum(mask)

                locs.append(
                    torch.hstack(
                        [
                            xsamps[mask].reshape(-1, 1),
                            ysamps[mask].reshape(-1, 1),
                        ]
                    )
                )

            self.scan_locs = torch.vstack(locs)[:num_scans, :]
        elif conf["scan_locs_type"] == "trajectory":
            # TODO: copy over from other thing
            pass

        # Generate the densities
        coarse_pts, beams, coarse_dists = self.nenv.lidar.generate_scan_points(
            self.scan_locs, num_points=self.conf["num_course_data"]
        )

        if conf["importance_samp_data"]:
            # TODO: Optimize this process so that we only compute the
            # density for the course points once.
            B, n_b, n_p, _ = coarse_pts.shape

            # compute course densities
            course_rho = self.nenv.get_densities(
                coarse_pts, opacity=conf["opacity"]
            )

            # Compute relevance weights
            _, _, coarse_weights = nerf2d_utils.depth_from_densities(
                coarse_dists, course_rho
            )

            # Sample bins based on weights
            fine_inds = nerf2d_utils.fine_sample(
                coarse_weights, conf["num_fine_data"]
            )
            # add points to sampled bins
            fine_dists = torch.gather(
                coarse_dists, -1, fine_inds
            ) + self.nenv.lidar.bin_length * torch.rand(fine_inds.shape).to(
                device
            )

            # concatenate coarse + fine points together
            all_points = torch.cat(
                (
                    coarse_pts,
                    self.scan_locs.reshape(B, 1, 1, 2)
                    + fine_dists.reshape(B, n_b, -1, 1) * beams,
                ),
                axis=-2,
            )

            # concat. dists
            all_dists = torch.cat((coarse_dists, fine_dists), axis=-1)

            # sort points by distance so they're ordered
            all_dists, inds = all_dists.sort()

            # sort point locations accordingly
            all_points = torch.gather(
                all_points, -2, inds.unsqueeze(-1).expand(-1, -1, -1, 2)
            )

        else:
            # just use the coarse points
            all_points = coarse_pts
            all_dists = coarse_dists

        all_rho = self.nenv.get_densities(all_points, opacity=conf["opacity"])

        d_target, _, all_weights = nerf2d_utils.depth_from_densities(
            all_dists, all_rho
        )

        w_mask = torch.sum(all_weights, -1) > conf["beam_thresh"]

        self.tensor_dataset = torch.utils.data.TensorDataset(
            self.scan_locs, d_target, w_mask
        )

    def __getitem__(self, idx):
        return self.tensor_dataset[idx]

    def __len__(self):
        return len(self.tensor_dataset)