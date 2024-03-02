import numpy as np
import torch.nn as nn
import torch
import pickle as pick
def ceil_ste(x):
    return (x.ceil() - x).detach() + x
class StraightThrough(nn.Module):
    """

    """
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class En_Decoding2(nn.Module):
    def __init__(self, sim_length: int,channel_num: int = 1):
        super().__init__()

        # Initialization of neuron parameters on CIFAR100
        # self.sita = torch.nn.Parameter(torch.FloatTensor([4.5]), requires_grad=True)
        # self.t_d = torch.nn.Parameter(torch.FloatTensor([-1.05]), requires_grad=True)
        # self.k = torch.nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)
        # self.sita_2 = torch.nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        # self.t_d_2 = torch.nn.Parameter(torch.FloatTensor([-0.5]), requires_grad=True)
        # self.k_2 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # Initialization of neuron parameters on CIFAR10
        self.sita = torch.nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.t_d = torch.nn.Parameter(torch.FloatTensor([-0.9]), requires_grad=True)
        self.k = torch.nn.Parameter(torch.FloatTensor([0.9]), requires_grad=True)
        self.sita_2 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.t_d_2 = torch.nn.Parameter(torch.FloatTensor([-0.8]), requires_grad=True)
        self.k_2 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.sim_length = sim_length
        self.num = None
        self.num2 = None

    def en_decoding(self, input: torch.Tensor):

        t = torch.relu(ceil_ste(((- torch.log(torch.relu(input / self.sita) + 1e-20)) + self.t_d) / self.k))

        t_max = torch.full(t.size(), self.sim_length).cuda()

        t = torch.where(t>self.sim_length, t_max.float()+5, t)

        u_d = torch.exp((-self.k * t + self.t_d) ) * self.sita

        u_d_min = u_d.min()
        zero_fill_ud = torch.zeros_like(u_d).cuda()
        u_d = torch.where(u_d == u_d_min,zero_fill_ud,u_d)


        t_2 = torch.relu(ceil_ste(
            ((-torch.log(torch.relu((input - u_d) / self.sita_2) + 1e-20)) + self.t_d_2) / self.k_2))

        t_2 = torch.where(t_2 > self.sim_length, t_max.float() , t_2)

        t_2 = torch.where(t_2 <= t, t, t_2)

        u_d_2 = torch.exp((-self.k_2 * t_2 + self.t_d_2)) * self.sita_2

        u_d_2_min = u_d_2.min()
        zero_fill = torch.zeros_like(u_d_2).cuda()

        u_d_2 = torch.where(u_d_2 == u_d_2_min, zero_fill, u_d_2)

        u_final = u_d + u_d_2

        # o = t.numel()
        # self.num = o

        # Count the number of spikes, Y is the encoding spike, Z is the calibrating spike
        # y = torch.nonzero(u_d > 0)
        # self.num = y.shape[0]
        # z = torch.nonzero(u_d_2 > 0)
        # self.num2 = z.shape[0]

        return u_final,self.num,self.num2

    def forward(self, input: torch.Tensor):
        u_d,t,t1 = self.en_decoding(input)
        return u_d

class AvgPoolConv(nn.Conv2d):
    """
    Converting the AvgPool layers to a convolution-wrapped module,
    so that this module can be identified in Spiking-refactor.
    """
    def __init__(self, kernel_size=2, stride=2, input_channel=64, padding=0, freeze_avg=True):
        super().__init__(input_channel, input_channel, kernel_size, padding=padding, stride=stride,
                         groups=input_channel, bias=False)
        # init the weight to make them equal to 1/k/k
        self.set_weight_to_avg()
        self.freeze = freeze_avg
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        self.set_weight_to_avg()
        x = super().forward(*inputs)
        return self.relu(x)

    def set_weight_to_avg(self):
        self.weight.data.fill_(1).div_(self.kernel_size[0] * self.kernel_size[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
