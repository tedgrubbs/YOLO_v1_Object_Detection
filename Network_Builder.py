
import torch

class Network_Builder(torch.nn.Module):
    def __init__(self, config):
        super(Network_Builder, self).__init__()

        self.cfg = config
        img_channels = self.cfg['input_shape'][1]
        img_height = self.cfg['input_shape'][2]
        img_width = self.cfg['input_shape'][3]

        network_description = self.cfd['main_net']

        # stores the actual network layers for pytorch to use later
        self.network = torch.nn.ModuleList()

        # records data regarding each layer to build subsequent layers
        layers = []
