import torch


class NeRF2DDataloader(torch.utils.data.Dataset):
    """When generating NeRF data this allows us
    to pre-compute the ground truth data for scans
    from a list of poses in a 2D image.

    Args:
        torch ([type]): [description]
    """

    def __init__(self, renderer) -> None:
        super().__init__()

        self.renderer = renderer
