# Usage: pt_pqt.py --path models/<modelname>
import torch
import torchvision
import os
import torch.nn as nn
import argparse
import time
from PIL import Image
from simple_network import MyNet

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path of the jit model")
args = vars(ap.parse_args())

batch_size_train = 64
batch_size_test = 64

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

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
test_ = iter(test_loader)
images, labels = next(test_)
print("shape {} ".format(images.shape))

print("[INFO] predicting...")

data = images[0]
label = labels[0]

data = torch.unsqueeze(data, dim=0)
print("data.shape {} ".format(data.shape))
data.to(cpu_device)
model.to(cpu_device)
print("after unsqueeze data.shape {} ".format(data.shape))
print("data.shape {} ".format(data.shape))

start = time.time()
output = model(data)
end = time.time()
print("Elapsed time = {} ms", (end - start) * 1000)
# max returns (value ,index)
_, predicted = torch.max(output.data, 1)

print("output.shape: {}", predicted.shape)
print("preds: {}", predicted.item())
print("label: {}", label.item())
