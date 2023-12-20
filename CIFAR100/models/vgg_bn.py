import torch
import torchvision
import torch.nn as nn

class VGGEncoder(nn.Module):
  def __init__(self):
    super(VGGEncoder, self).__init__()
    self.network = torchvision.models.vgg16_bn(pretrained=True)
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
  def forward(self, x):
    return self.avg_pool(self.network.features(x))