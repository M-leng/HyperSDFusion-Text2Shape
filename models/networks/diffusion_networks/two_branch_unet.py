import torch
import torch.nn as nn
from .openai_model_3d import UNet3DModel
from models.networks.diffusion_networks.ldm_diffusion_util import (
    conv_nd,
    zero_module,
    normalization,
)
from torch_geometric.data import Data, Batch

class TwoBranchDiffusionUNet(nn.Module):
    def __init__(self, unet_params, vq_conf=None, conditioning_key=None):
        """ init method """
        super().__init__()

        self.base_branch = UNet3DModel(**unet_params)
        self.shape_branch = UNet3DModel(**unet_params)
        self.conditioning_key = conditioning_key # default for lsun_bedrooms
        self.branchtwo = True

        self.out = nn.Sequential(
            normalization(unet_params.model_channels*2),
            nn.SiLU(),
            zero_module(conv_nd(3, unet_params.model_channels*2, 3, 3, padding=1)),
        )



    def forward(self, x, t, c_base: list = None, c_graph: list = None):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)

        if self.conditioning_key is None:
            out = self.diffusion_net(x, t)
        elif self.conditioning_key == 'crossattn':
            cc_base = torch.cat(c_base, 1)
            cc_graph = torch.cat(c_graph, 1)
            return_dis = False
            return_feature = False

            if return_dis:
                d1, d2, d3 = self.base_branch(x, t, context=cc_base)
                d11, d22, d33 = self.shape_branch(x, t, context=cc_graph)
                return (d1+d11)/2., (d2+d22)/2., (d3+d33)/2.
            if return_feature:
                d1, d2, d3 = self.base_branch(x, t, context=cc_base)
                d11, d22, d33 = self.shape_branch(x, t, context=cc_graph)
                return d11, d22, d33
            out_base, hyper_loss1 = self.base_branch(x, t, context=cc_base)
            out_graph, hyper_loss2 = self.shape_branch(x, t, context=cc_graph)

            out = torch.cat([out_base, out_graph], dim=1)
            out = self.out(out)
        else:
            raise NotImplementedError()

        hyper_loss = hyper_loss1 + hyper_loss2
        return out, hyper_loss


