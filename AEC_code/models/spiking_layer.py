import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple, List, Union, Dict, cast
from torch.utils.data import DataLoader
from models.utils import StraightThrough, AvgPoolConv
# from models.data_get import GetLayerInputOutput
#脉冲模块里面定义了脉冲模式和非脉冲模式，分别对应CNN和SNN的输入与输出
class SpikeModule(nn.Module):
    """
    Spike-based Module that can handle spatial-temporal information.
    去除了输入移位
    """
    def __init__(self, sim_length: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = False):
        super(SpikeModule, self).__init__()
        if isinstance(conv, nn.Conv2d):
            self.fwd_kwargs = {"stride": conv.stride, "padding": conv.padding,
                               "dilation": conv.dilation, "groups": conv.groups}
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
        self.threshold = None#阈值的设置？是否归一化？
        self.outw = 1.0
        self.mem_pot = 0
        self.input_shift = None
        self.mem_pot_init = 0
        self.weight = conv.weight
        self.org_weight = copy.deepcopy(conv.weight.data)
        if conv.bias is not None:
            self.bias = conv.bias
            self.org_bias = copy.deepcopy(conv.bias.data)
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the spike forward default
        self.use_spike = False
        self.enable_shift = enable_shift
        self.sim_length = sim_length
        self.cur_t = 0
        self.relu = StraightThrough()
#非脉冲模式走relu，脉冲模式走膜电位
    def forward(self, input: torch.Tensor):
        if self.use_spike:
            self.cur_t += 1
            if self.input_shift is not None:
                self.input_shift = self.input_shift.mean(0, keepdim=True)
                self.input_shift = self.input_shift.repeat_interleave(input.size(0), dim=0)
                input = input + self.input_shift
            x = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
            if self.enable_shift is True and self.threshold is not None:
                x = x + self.threshold * 0.5 / self.sim_length
            self.mem_pot = self.mem_pot + x
            spike = (self.mem_pot >= self.threshold).float() * (self.outw * self.threshold)
            self.mem_pot -= spike
            return spike
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs))

    def init_membrane_potential(self):
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.cur_t = 0

class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.spike_module_refactor(self.model, sim_length)
        self.use_spike = False
        self.converge = None
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

    def set_spike_state(self, use_spike: bool = True):
        self.use_spike = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.use_spike = use_spike

    def init_membrane_potential(self):
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.init_membrane_potential()

    def forward(self, input):
        if self.use_spike:
            self.init_membrane_potential()
            converge = []
            out = 0
            for sim in range(self.T):
                out = out + self.model(input)
                acc = out.data.detach()
                converge.append(acc)
            self.converge = converge
        else:
            out = self.model(input)
        return out,self.converge

#后续代码：获取最大脉冲，收敛时间，分布的精度

#初始化阈值

class DataSaverHook:
    def __init__(self,percentile:Union[float,None] = None):
        self.max_act = None
        self.percentile = percentile
    def __call__ (self, module, input_batch, output_batch):
        def get_act_thresh(tensor):
            if self.percentile is not  None:
                assert 0. <= self.percentile <= 1.0
                act_thresh = quantile(output_batch,self.percentile)
            else:
                act_thresh = tensor.max()
            return act_thresh
        if self.max_act is None:
            self.max_act = get_act_thresh(output_batch)
        else:
            cur_max = get_act_thresh(output_batch)
            self.max_act = self.max_act if self.max_act > cur_max else cur_max
        module.threshold = self.max_act


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p*100)).type_as(tensor)

@torch.no_grad()
def set_init_threshold (train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           percentile: Union[float, None] = None,
                           ):

    # do not use train mode here (avoid bn update)
    model.set_spike_state(use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    hook_list = []
    for m in model.modules():
        if isinstance(m, SpikeModule):
            hook_list += [m.register_forward_hook(DataSaverHook(percentile))]
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        _ = model(input)
    for h in hook_list:
        h.remove()

