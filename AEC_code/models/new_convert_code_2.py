import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple, List, Union, Dict, cast
from models.utils import StraightThrough, AvgPoolConv,En_Decoding,En_Decoding2,En_Decoding3

#权重归一化
def ceil_ste(x):
    return (x.ceil() - x).detach() + x

class SpikeModule(nn.Module):
    def __init__(self, sim_length: int, conv: Union[nn.Conv2d, nn.Linear]):
        super(SpikeModule, self).__init__()
        if isinstance(conv, nn.Conv2d):
            self.fwd_kwargs = {"stride": conv.stride, "padding": conv.padding,
                               "dilation": conv.dilation, "groups": conv.groups}
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear

        self.weight = conv.weight
        self.org_weight = copy.deepcopy(conv.weight.data)

        if conv.bias is not None:
            self.bias = conv.bias
            self.org_bias = copy.deepcopy(conv.bias.data)
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the spike forward default
        self.use_spike = True

        self.sim_length = sim_length



    def forward(self,input:torch.Tensor):
        return self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)

class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.sim_length = sim_length
        self.spike_module_refactor(self.model, sim_length)

        assert sim_length > 0, "SNN does not accept negative simulation length"



    def spike_module_refactor(self, module: nn.Module, sim_length: int, prev_module=None):
        """
        Recursively replace the normal conv2d to SpikeConv2d
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param sim_length: simulation length, aka total time steps
        :param prev_module: use this to add relu to prev_spikemodule
        """
        for name, immediate_child_module in module.named_children():
            new = En_Decoding2(sim_length=sim_length)
            # new = StraightThrough()
            if isinstance(immediate_child_module,nn.ReLU):
                setattr(module, name, new)
            self.spike_module_refactor(immediate_child_module,sim_length)
    def forward(self, input):
        out = self.model(input)
        return out



























