from ...Blocks.Basic import CoBaRe, CoSigUp

import torch.nn as nn


class FF(nn.Module):
    def __init__(self, cfgs):
        super(FF, self).__init__()
        self._make_layers(cfgs)
        self.cfgs = cfgs
        self.sigmoid_head = nn.Sigmoid()

    def forward(self, x):
        for k, _ in self.cfgs["mid"].items():
            x = getattr(self, k)(x)
        
        # segment head
        x = getattr(self, "head")(x)
        # sigmoid head
        if self.cfgs["sigmoid_head"]:
            x = self.sigmoid_head(x)
        return x
    
    def _make_layers(self, cfgs):
        for k, v in cfgs['mid'].items():
            self.add_module(k, CoBaRe(**v))

        head_args = cfgs["head"]
        self.add_module("head", nn.Conv2d(**head_args))


    
def get_configs(in_ch = 3, out_ch = 1, mid_ch = 32, kernel_size = 3):
    return {
        "mid":
            {
                "ff1": dict(kernel_size = kernel_size, in_ch = in_ch, out_ch = mid_ch, padding = "same"),
                "ff2": dict(kernel_size = kernel_size, in_ch = mid_ch, out_ch = mid_ch, padding = "same"),
                "ff3": dict(kernel_size = kernel_size, in_ch = mid_ch, out_ch = mid_ch, padding = "same"),
                "ff4": dict(kernel_size = kernel_size, in_ch = mid_ch, out_ch = mid_ch, padding = "same"),
            }, 
        "head":
            dict(
                kernel_size = kernel_size, in_channels = mid_ch, out_channels = out_ch, padding = "same"
            ),
        "sigmoid_head": False,
    }
def get_ff(in_ch = 3, out_ch = 1, mid_ch = 32, kernel_size = 3):
    return FF(cfgs = get_configs(in_ch = in_ch, out_ch = out_ch, mid_ch = mid_ch, kernel_size = kernel_size))

