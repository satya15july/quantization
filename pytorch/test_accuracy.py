import torch
import torchvision
import os
import torch.nn as nn
import argparse
import time
from simple_network import MyNet

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path of the jit model")
args = vars(ap.parse_args())

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

n_epochs = 3
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))
cpu_device = "cpu"

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

model = MyNet(numChannels=1, classes=10).to(cpu_device)
path = args["path"]
model = torch.jit.load(path, map_location="cpu")
print(model)
model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(cpu_device)
        labels = labels.to(cpu_device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print('Accuracy of the model {} on the 10000 test images: {} '.format(path, acc))