import torch
import torchvision
import torch.nn as nn

class ResnetEncoder(nn.Module):
  def __init__(self):
    super(ResnetEncoder, self).__init__()
    self.network = torchvision.models.resnet152(pretrained=False)
    num_ftrs = self.network.fc.in_features
    self.network.fc = nn.Linear(num_ftrs, 100)

  def forward(self, x):
    return self.network(x)