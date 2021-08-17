import architectures
import meta_learners
import numpy as np
import torch
import torch_interpolations

from hydra import experimental
from matplotlib import pyplot as plt
from skimage import filters
from utils import lidar
from utils import models
from utils import utils

l1_loss = lambda x, y: torch.abs(x - y)
l2_loss = lambda x, y: 1e3*torch.square(x - y)

def train(verbose=True):
    """
    Trains a meta-learner on the 2D MNIST toy problem.
    
    Args:
        cfg.arch_type:               architecture type. selects how the context 
                              is incorporated into the decoder.
        cfg.learner_type:               learner type. selects which meta-learning 
                              method to use.
        cfg.num_context:         number of context points shown to 
                              the meta-learner.
        loss_type:            picks between l1 and l2 loss functions.
        CNN_partial_context:  flags if CNN should see only half the edge set.
        verbose:              flag for printing.
    """
    
    # open current config file with hydra
    with experimental.initialize(config_path="config"):
        cfg = experimental.compose(config_name="single_config")
        
        # select device
        if cfg.use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        dtype = torch.float32

        # select loss
        if cfg.loss_type == "l1":
            loss_func = l1_loss
        else:
            loss_func = l2_loss
            
        # select activation
        if cfg.activation == "tanh":
            activation = torch.nn.Tanh()
        else:
            activation = torch.nn.ReLU()
            
        # setup model
        model = models.FFNet(cfg.net_shape,
                             activation=activation).to(device, dtype)
        
        if cfg.fourier_feats:
            # setup Fourier features
            c = np.sqrt(6/2)
            embed = torch.nn.Parameter(
                cfg.w0 * torch.FloatTensor(
                1, 1, 2, cfg.num_feats).uniform_(-c, c).to(
                device, dtype), requires_grad=True)
#             embed = torch.nn.Parameter(cfg.embed_std*torch.rand(
#                 1, 1, 2, cfg.num_feats).to(
#                 device, dtype), requires_grad=True)

            # collect params to be optimized
            all_params = torch.nn.ParameterList(
                list(model.parameters()) + [embed])
        
        else:
            all_params = model.parameters()

        opt = torch.optim.Adam(all_params, lr=cfg.lr)

        # load MNIST
        x_train, y_train, x_val, y_val = utils.load_data()
        losses = []

        # load image onto GPU
        img = torch.from_numpy(x_train[cfg.train_ind].reshape(
            28,28).T[:,::-1].copy()).to(device, dtype)
        imgs = img.unsqueeze(0).expand(cfg.batch_size, -1, -1)
        
        # create interpolation instance
        pts = torch.linspace(-1, 1, 28).to(device, dtype)
        gi = torch_interpolations.RegularGridInterpolator((pts,pts), img)
        
        # create Lidar
        li = lidar.Lidar(t_n=cfg.t_n, t_f=cfg.t_f, num_beams=cfg.num_beams)
        bin_length = (li.t_f - li.t_n) / cfg.num_beams
        
        # if not resampling lidar points, create fixed view points
        if not cfg.sample_views:
            
            # if viewed from only one angle
            if cfg.single_view:
                theta = 2*np.pi*torch.rand(1).to(device, dtype)
            
            else:
                theta = 2*np.pi*torch.rand(cfg.batch_size).to(device, dtype)
            
            # create points on circle
            x = torch.stack((torch.cos(theta), torch.sin(theta)), 
                                  dim=1).expand(cfg.batch_size, -1)
            
            # map circle points onto square
            scales = torch.max(torch.abs(x), 1, keepdim=True)[0]
            x = cfg.r * x / scales

        # training loop
        for tt in range(cfg.train_iters):
            opt.zero_grad()
            
            # if resampling lidar origins each time
            if cfg.sample_views:
                theta = 2*np.pi*torch.rand(cfg.batch_size).to(device, dtype)
            
                x = torch.stack((torch.cos(theta), torch.sin(theta)), 
                                      dim=1).expand(cfg.batch_size, -1)
                
                # map circle points onto square
                scales = torch.max(torch.abs(x), 1, keepdim=True)[0]
                x = cfg.r * x / scales
            
            # create lidar scan
            coarse_points, beams, coarse_dists = li.generate_scan_points(
            x, num_points=cfg.num_coarse)
            
            # if resampling fine points
            if cfg.importance_sample:
                
                _, n_b, n_p, _ = coarse_points.shape
            
                # get point densities from ground truth
                rho = cfg.opacity * torch.stack([gi(
                    (coarse_points[ii, :, :, 0].reshape(-1), 
                     coarse_points[ii, :, :, 1].reshape(-1))).reshape(n_b, n_p)
                     for ii in range(cfg.batch_size)])

                # generate "coarse" termination probabilities
                _, _, coarse_weights = utils.depth_from_densities(
                    coarse_dists, rho)

                # sample bins based on their weights
                fine_inds = utils.fine_sample(coarse_weights, 
                                              num_samples=cfg.num_fine)

                # add points to sampled bins
                fine_dists = (torch.gather(coarse_dists, -1, fine_inds) 
                          + bin_length*torch.rand(fine_inds.shape
                                                 ).to(device, dtype))

                # concatenate coarse + fine points together
                all_points = torch.cat(
                    (coarse_points, x.reshape(cfg.batch_size,1,1,2) 
                    + fine_dists.reshape(cfg.batch_size, n_b, -1, 1) 
                    * beams), axis=-2)

                # concat. dists
                all_dists = torch.cat((coarse_dists, fine_dists), axis=-1)

                # sort points by distance so they're ordered
                all_dists, inds = all_dists.sort()
                
                # sort point locations accordingly
                all_points = torch.gather(
                    all_points, -2, inds.unsqueeze(-1).expand(-1, -1, -1, 2))

            else:
                # just use the coarse points
                all_points = coarse_points
                all_dists = coarse_dists
            
            # get point densities from ground truth
            _, n_b, n_p, _ = all_points.shape
    
            rho = cfg.opacity * torch.stack([gi(
                (all_points[ii, :, :, 0].reshape(-1), 
                 all_points[ii, :, :, 1].reshape(-1))).reshape(n_b, n_p)
                for ii in range(cfg.batch_size)])
        
            d_target, var_target, all_weights = utils.depth_from_densities(
                all_dists, rho)
            
            if cfg.fourier_feats:
                feats = torch.sin(torch.matmul(all_points, embed))
            else:
                feats = all_points
                
            rho_hat = torch.exp(model(feats)).squeeze(-1)
            d_mean, d_var, _ = utils.depth_from_densities(all_dists, rho_hat)
            
            # mask beams with total density < beam_thresh
            w_mask = (torch.sum(all_weights, -1) > cfg.beam_thresh)
            
            # compute loss, weighted by variance
            loss = torch.mean(w_mask * loss_func(d_mean, d_target)
                              / torch.sqrt(d_var.detach() + 1e-2))

            # backprop
            loss.backward()

            print(loss)

            losses.append(loss.cpu().detach().numpy())
            opt.step()

        # sample test points (all pixels in image)
        nx = ny = 100
        pix = utils.region_pixels((0.,0.), 2, 2, nx, ny
                                 ).T.reshape(1, -1, 2).repeat(1, axis=0)
        
        pix_points = torch.from_numpy(pix).to(device, dtype)
        
        if cfg.fourier_feats:
            pix_feats = torch.sin(torch.matmul(pix_points, embed))
        else:
            pix_feats = pix_points
            
        rho_hat = torch.exp(model(pix_feats)).squeeze(-1)
        # Plot true / estimated density
        fig, axs = plt.subplots(2,2,figsize=(10,10))
    
        im = axs[0,0].imshow(rho_hat.detach().cpu().numpy().reshape(nx,ny)[::-1,:])
        axs[0,0].set_title("Estimated")
        
        axs[0,1].imshow(x_train[cfg.train_ind].reshape(
            cfg.IMG_SIZE, cfg.IMG_SIZE))
        axs[0,1].set_title("Ground Truth")
        
        num_plot = min(32, cfg.batch_size)
        if cfg.sample_views:
            theta = 2*np.pi*torch.rand(num_plot).to(device, dtype)
            
            x = torch.stack((torch.cos(theta), torch.sin(theta)), 
                                  dim=1).expand(num_plot, -1)
            # map circle points onto square
            scales = torch.max(torch.abs(x), 1, keepdim=True)[0]
            x = cfg.r * x / scales
            
        x_test = torch.atleast_2d(x[:num_plot, :])
        coarse_points, beams, coarse_dists = li.generate_scan_points(
            x_test, num_points=cfg.num_coarse)
        
        if cfg.importance_sample:

            _, n_b, n_p, _ = coarse_points.shape

            # get point densities from ground truth
            rho = cfg.opacity * torch.stack([gi(
                (coarse_points[ii, :, :, 0].reshape(-1), 
                 coarse_points[ii, :, :, 1].reshape(-1))).reshape(n_b, n_p)
                 for ii in range(cfg.batch_size)])

            _, _, coarse_weights = utils.depth_from_densities(
                coarse_dists, rho)

            fine_inds = utils.fine_sample(coarse_weights, 
                                          num_samples=cfg.num_fine)
            
            fine_dists = (torch.gather(coarse_dists, -1, fine_inds) 
                          + bin_length*torch.rand(fine_inds.shape
                                                 ).to(device, dtype))

            all_points = torch.cat(
                (coarse_points, x.reshape(cfg.batch_size,1,1,2) 
                + fine_dists.reshape(cfg.batch_size, n_b, -1, 1) 
                * beams), axis=-2)
    
            all_dists = torch.cat((coarse_dists, fine_dists), axis=-1)

            all_dists, inds = all_dists.sort()
            all_points = torch.gather(
                all_points, -2, inds.unsqueeze(-1).expand(-1, -1, -1, 2))
            
        else:
            all_points = coarse_points
            all_dists = coarse_dists

            
        # get point densities from ground truth
        rho = cfg.opacity* (torch.stack([gi((all_points[ii, :, :, 0], 
                                             all_points[ii, :, :, 1])) 
                            for ii in range(num_plot)]))
        d_target, _, all_weights = utils.depth_from_densities(all_dists, rho)
        w_mask = (torch.sum(all_weights, -1) > cfg.beam_thresh).cpu()
    
        hit_points = (x_test.reshape(num_plot, 1, 2) + beams.squeeze(-2) 
                      * d_target.reshape(num_plot,-1,1))
        
        s = axs[1,0].scatter(all_points[:,:,:,0].cpu(), all_points[:,:,:,1].cpu(), c=rho.cpu())
        axs[1,0].scatter(w_mask*hit_points[:,:,0].cpu(), w_mask*hit_points[:,:,1].cpu(), c='C3')
        axs[1,0].scatter(x_test[:,0].cpu(), x_test[:,1].cpu(), c='C3')
        axs[1,0].set_title("True Scan")
        plt.axis("equal")
        
        if cfg.fourier_feats:
            feats = torch.sin(torch.matmul(all_points, embed))
        else:
            feats = all_points

        rho_hat = torch.exp(model(feats)).squeeze(-1)
        d_mean, d_var, _ = utils.depth_from_densities(all_dists, rho_hat)
        
        hat_points = (x_test.reshape(num_plot, 1, 2) + beams.squeeze() 
                      * d_mean.reshape(num_plot,-1,1))
        
        s = axs[1,1].scatter(all_points[:,:,:,0].cpu(), 
                             all_points[:,:,:,1].cpu(), 
                             c=rho_hat.cpu().detach())
        axs[1,1].scatter(w_mask*hat_points[:,:,0].cpu().detach(), 
                         w_mask*hat_points[:,:,1].cpu().detach(), 
                         c='C1', label="est")
        axs[1,1].scatter(w_mask*hit_points[:,:,0].cpu().detach(), 
                         w_mask*hit_points[:,:,1].cpu().detach(), 
                         c='C3', label="true")
        
        plt.legend()
        axs[1,1].set_title("Est Scan")
        plt.axis("equal")
        

        return model
