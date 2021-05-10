import os
from torchvision import transforms
import torch
from PIL import Image
from skimage import feature
import scipy.ndimage
import numpy as np
import random


class SDFTransform:
    """This defines a new torchvision transform to convert BW images to
    SDFs, and then load it into a torchvision.transforms. Compose pipeline.
    An SDF is negative outside the set and positive inside the set.
    """

    def __call__(self, img):
        # Threshhold the incoming image
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0
        img_bin = img.squeeze(0).numpy()

        # Find the edges of the image
        img_edge = np.logical_not(feature.canny(img_bin))

        # Compute the distances to the edges
        sdf_grid = scipy.ndimage.morphology.distance_transform_edt(img_edge)

        # Negate the values "inside" the digit
        sdf_grid[img_bin == 1.0] *= -1.0
        sdf_grid /= float(img.shape[0])
        sdf_grid = torch.Tensor(sdf_grid)

        return sdf_grid


def cubi_preprocess():
    # Parameters
    source_dir = "../data/cubi_sdf/out_empty/"
    target_dir = "../data/cubi_sdf/sl512_preproc/"
    overwrite = True
    small_sidelen = 512

    # Make the target directory
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    # Compose the transform
    tf = transforms.Compose(
        [
            transforms.Resize(small_sidelen),
            transforms.ToTensor(),
            SDFTransform(),
        ]
    )

    fnames = os.listdir(source_dir)

    if os.path.isfile(os.path.join(target_dir, "pzcounts.pt")):
        pzcounts = torch.load(os.path.join(target_dir, "pzcounts.pt"))
    else:
        pzcounts = {}

    for (i, name) in enumerate(fnames):
        name_str = os.path.splitext(name)[0] + ".pt"
        save_pth = os.path.join(target_dir, name_str)

        if os.path.isfile(save_pth) and not overwrite:
            continue

        img = Image.open(os.path.join(source_dir, name))
        sdf = tf(img)
        torch.save(sdf, save_pth)

        sdf_vals = sdf.reshape((-1, 1))
        zero_inds = (torch.nonzero(sdf_vals == 0.0, as_tuple=False))[:, 0]

        nzeros = zero_inds.shape[0] 
        pzcounts[name_str] = {"npixels": sdf_vals.shape[0], "nzeros": nzeros}

        if (i + 1) % 100 == 0:
            print("Progress: ", i, " / ", len(fnames))

    # Train/test split
    delim = int(len(pzcounts.keys()) * 0.9)
    ks = list(pzcounts.keys())
    random.shuffle(ks)
    split_sets = {"train": ks[:delim], "test": ks[delim:]}

    # Save counts and split set
    torch.save(split_sets, os.path.join(target_dir, "split_sets.pt"))
    torch.save(pzcounts, os.path.join(target_dir, "pzcounts.pt"))


if __name__ == "__main__":
    cubi_preprocess()
