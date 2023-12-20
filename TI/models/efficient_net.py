import torch
import torchvision
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
  def __init__(self):
    super(EfficientNetClassifier, self).__init__()
    self.network = torchvision.models.efficientnet_v2_s(pretrained=True)
    # num_ftrs = self.network.fc.in_features
    num_ftrs = self.network.classifier[1].in_features
    self.resize = torchvision.transforms.Resize(224)
    self.network.classifier[1] = nn.Linear(num_ftrs, 200)

  def forward(self, x):
    # returns logits
    return self.network(self.resize(x))