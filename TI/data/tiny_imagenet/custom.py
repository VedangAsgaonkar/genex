from tiny_imagenet import TinyImageNet
import torchvision, torchvision.transforms as transforms 

transform = transforms.Compose([
    transforms.ToTensor(),
    
])

dataset = TinyImageNet(root = "./data/tiny_imagenet/", download = False, transform = transform)
# mean, std = dataset[0].mean([1,2]), dataset[0].std([1,2])
print(type(dataset))