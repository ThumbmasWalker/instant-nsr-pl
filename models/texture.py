import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.dir_encoding_config['n_dir_dims']
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    
    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)

        '''
		xyz = features[int:int]

		dirs_embd = get_encoding_nets(xyz, dirs)

		def get_encoding_nets(self, xyz, dirs):

			encoding_dict = { xyz_1: tcnn.Encoding, xyz_2:tcnn.Encoding, ...}

			for key, val in encoding_dict:

				find nets, xyzs and interpolation coefficients 

				How to do this efficiently like in SHE?
		
			feats = nets(dirs)
			
			interpolated_feat = interpolation_coeffs*nets

		
		
		'''
        if self.n_dir_dims == 2:
            x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
            theta = torch.atan2(y, x)
            r_xy = torch.sqrt(x**2 + y**2)
            phi = torch.atan2(r_xy, z)
            theta_scaled = (theta / (2 * torch.pi) + 1) % 1
            phi_scaled = (phi / torch.pi) % 1
            dirs = torch.cat([theta_scaled.unsqueeze(dim=1), phi_scaled.unsqueeze(dim=1)], dim=1)

        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}




@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
