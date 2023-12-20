import torch
import torchvision
import torch.nn as nn

class VitBClassifier(nn.Module):
  def __init__(self):
    super(VitBClassifier, self).__init__()
    self.network = torchvision.models.resnet18(pretrained=True)
    # self.network = torchvision.models.swin_b(pretrained=True)
    # num_ftrs = self.network.fc.in_features
    # num_ftrs = self.network.classifier[1].in_features
    # self.network.classifier[1] = nn.Linear(num_ftrs, 200)
    # num_ftrs = self.network.heads.head.in_features
    # self.network.heads.head = nn.Linear(num_ftrs, 200)
    # self.network.head = torch.nn.Linear(1024 , 200)
    self.network.fc = nn.Linear(512, 200, bias = True)

  def forward(self, x):
    # returns logits
    return self.network(x)