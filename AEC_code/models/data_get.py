import torch
from torch.utils.data import DataLoader
from models.spiking_layer import SpikeModule, SpikeModel
import numpy as np
from typing import Callable, Tuple, List, Union, Dict, cast



class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """
    def __init__(self):
        self.stored_output = None
        self.stored_input = None
        #residual用于残差网络
        # self.stored_residual = None
    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch
        else:
            self.stored_output = output_batch + self.stored_output
        if self.stored_input is None:
            self.stored_input = input_batch[0]
        else:
            self.stored_input = input_batch[0] + self.stored_input
        # if len(input_batch) == 2:
        #     if self.stored_residual is None:
        #         self.stored_residual = input_batch[1].detach()
        #     else:
        #         self.stored_residual = input_batch[1].detach() + self.stored_residual
        # else:
        #     if self.stored_residual is None:
        #         self.stored_residual = 0

    def reset(self):
        self.stored_output = None
        self.stored_input = None
        # self.stored_residual = None


class GetLayerInputOutput:
    def __init__(self, model: SpikeModel, target_module: SpikeModule):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        # do not use train mode here (avoid bn update)
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        # note that we do not have to check model spike state here,
        # because the SpikeModel forward function can already do this
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), self.data_saver.stored_output.detach()
               # self.data_saver.stored_residual
def fittness():
    pass

def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p*100)).type_as(tensor)

# 按百分位初始化阈值
@torch.no_grad()
def set_init_threshold_percentile (train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           percentile: Union[float, None] = None,
                           iter:int = 30):
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, SpikeModule):
            # print('\nAdjusting threshold for layer {}:'.format(name))
            get_out = GetLayerInputOutput(model, module)
            device = next(module.parameters()).device
            cached_batches = []
            for i,(images,targets) in enumerate(train_loader):
                # compute the original output
                images = images.to(device)
                model.set_spike_state(use_spike=False)
                _, cur_out = get_out(images)
                get_out.data_saver.reset()
                cached_batches.append(cur_out)
                if i > iter:
                    break

            cur_outs = torch.cat([x for x in cached_batches])
            threshold_per = quantile(cur_outs,percentile)
            module.threshold = threshold_per


def out_rate(iters,test_loader:torch.utils.data.DataLoader,model:SpikeModel):
    for name,module in model.named_modules():
        if isinstance(module, SpikeModule):
            snn_mean_rate = rate_validate(iters,test_loader,model,module)
            print('\nmean_rate of layer {}:'.format(name))
            print(snn_mean_rate)

@torch.no_grad()
def rate_validate(iters,test_loader:torch.utils.data.DataLoader,model:SpikeModel,module:SpikeModule):
    mean_rate =[]
    get_out = GetLayerInputOutput(model,module)
    for i,(images, targets) in enumerate(test_loader):
        # compute the original output
        images = images.cuda()
        model.set_spike_state(use_spike=True)
        _, cur_out = get_out(images)
        get_out.data_saver.reset()
        cur_out =  cur_out.cuda()
        cur_out = cur_out/model.T
        fire_rate = cur_out/module.threshold
        mean_fire_rate = fire_rate.mean()
        mean_rate.append(mean_fire_rate)
        if i>=iters:
            break
    mean_rate_all = torch.as_tensor(mean_rate).mean()
    return mean_rate_all



def floor_ste(x):
    return (x.floor() - x).detach() + x


