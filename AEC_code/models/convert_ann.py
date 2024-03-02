import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple, List, Union, Dict, cast
from models.utils import StraightThrough, AvgPoolConv

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
        self.tau = torch.nn.Parameter(torch.FloatTensor([6]), requires_grad=True)
        self.t_d = torch.nn.Parameter(torch.FloatTensor([-6]), requires_grad=True)
        self.k = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        # e = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.tau_d = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.t_d_d = torch.nn.Parameter(torch.FloatTensor([-30]), requires_grad=True)
        self.sita = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.sita_d = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.sim_length = sim_length

        self.relu = StraightThrough()

    def en_decoding(self,input:torch.Tensor):
        input = torch.relu(input / self.sita)
        t = torch.relu(ceil_ste(((-self.tau*torch.log(input + 0.000001))+self.t_d)))
        t_max = torch.full(t.size(),self.sim_length).cuda()
        t = torch.where(t > self.sim_length,t_max,t)
        t_min = torch.zeros_like(t).cuda()
        t = torch.where(t < 0,t_min,t)
        u_d = torch.exp(-(self.k*t-self.t_d)/self.tau) * self.sita
        return u_d

    def forward(self,input:torch.Tensor):
        u = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
        # u_d = self.en_decoding(u)
        return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs))
        # return u_d
class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.spike_module_refactor(self.model, sim_length)

        assert sim_length > 0, "SNN does not accept negative simulation length"
        self.T = sim_length

    def spike_module_refactor(self, module: nn.Module, sim_length: int, prev_module=None):
        """
        Recursively replace the normal conv2d to SpikeConv2d
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param sim_length: simulation length, aka total time steps
        :param prev_module: use this to add relu to prev_spikemodule
        """
        prev_module = prev_module
        for name, immediate_child_module in module.named_children():
            if type(immediate_child_module) in self.specials:
                setattr(module, name, self.specials[type(immediate_child_module)]
                                                        (immediate_child_module, sim_length=sim_length))
            elif isinstance(immediate_child_module, nn.Conv2d) and not isinstance(immediate_child_module, AvgPoolConv):
                setattr(module, name, SpikeModule(sim_length=sim_length, conv=immediate_child_module))
                prev_module = getattr(module, name)
            elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
                if prev_module is not None:
                    prev_module.add_module('relu', immediate_child_module)
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(immediate_child_module, AvgPoolConv):
                relu = immediate_child_module.relu
                setattr(module, name, SpikeModule(sim_length=sim_length, conv=immediate_child_module))
                getattr(module, name).add_module('relu', relu)
            else:
                prev_module = self.spike_module_refactor(immediate_child_module, sim_length=sim_length, prev_module=prev_module)

        return prev_module

    def forward(self, input):
        out = self.model(input)
        return out



























