from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class standard_block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(standard_block, self).__init__()
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    out = self.conv2d(x)
    conv_out = out
    out = self.bn(out)
    out = self.relu(out)
    return out,conv_out

class PLAIN(nn.Module):
  def __init__(self, conv_config, num_classes):
    super(PLAIN, self).__init__()
    self.name_to_ind = OrderedDict()
    self.ind_to_name = OrderedDict()
    self.meta_data = OrderedDict()
    self.name_to_next_name = OrderedDict()
    # channel0 = conv_config[0]
    # self.conv0 = nn.Conv2d(3,channel0,kernel_size=3, padding=1, bias=False)
    # self.bn0 = nn.BatchNorm2d(channel0)
    # last_n = channel0
    self.layers = nn.ModuleList()
    last_layer = 'input'
    last_n = None
    fsize = 32

    for i, n in enumerate(conv_config):

      if i == 1:
        self.layers.append(standard_block(conv_config[0],conv_config[1]))
        last_n=conv_config[1]
        name = 'conv%d' % len(self.name_to_ind)
        self.name_to_ind[name] = i
        self.ind_to_name[i] = name
        self.meta_data[name] = {'n': n, 'c': last_n, 'ksize': 3, 'fsize': fsize}
        self.name_to_next_name[last_layer] = name
        last_layer = name
        last_n = n
      elif i==0:
        continue
      elif i==len(conv_config)-1:
        continue
      else:
        if n == 'M':
          self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          fsize /= 2
        else:
          self.layers.append(standard_block(last_n, n))
          name = 'conv%d' % len(self.name_to_ind)
          self.name_to_ind[name] = i
          self.ind_to_name[i] = name
          self.meta_data[name] = {'n': n, 'c': last_n, 'ksize': 3, 'fsize': fsize}
          self.name_to_next_name[last_layer] = name
          last_layer = name
          last_n = n

    self.fc = nn.Linear(conv_config[-1], num_classes)

    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
        # m.bias.data.zero_()

  def forward(self, x):

    # x = F.relu(self.bn0(self.conv0(x)))
    for layer in self.layers:
      x = layer(x)
      if isinstance(layer, standard_block):
        x = x[0]

    x = x.mean(2).mean(2)
    x = self.fc(x)
    return x

def plain20(num_classes=10):
  return PLAIN(conv_config=[3,16, 16, 16, 16, 16, 16, 'M',
                            32, 32, 32, 32, 32, 32, 'M',
                            64, 64, 64, 64, 64, 64, 'M',64],
               num_classes=num_classes)
def plain20pr(num_classes=10):#3, 8, 8, 8, 8, 8, 16, 8, 24, 24, 24, 24, 24, 32, 64, 64, 64, 48, 16, 16
  return PLAIN(conv_config=[8, 8, 8, 8, 8, 16,  'M',
                            8,24, 24, 24, 24, 24, 'M',
                            32,64, 64, 64, 48, 16, 'M'],
               num_classes=num_classes)

if __name__ == '__main__':
    from torchsummary import torchsummary
    net = plain20pr()
    net.to('cuda')

    print(net)