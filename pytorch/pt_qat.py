# Usage : pt_pqt.py --path models/32_dnn.pth
import torch
import torchvision
import os
import torch.nn as nn
import argparse
import time
from copy import deepcopy
import numpy as np
from simple_network import MyNet

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch', default='x86_64', choices=['x86_64', 'arm64'], help='Use x86_64 for desktop cpu.Otherwise, use arm64 for raspi device')
ap.add_argument("-p", "--path", required=True,
                help="path of the model")
args = vars(ap.parse_args())

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

cpu_device = "cpu"
batch_size_train = 64
batch_size_test = 64

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

test_set = torchvision.datasets.MNIST(root, train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))

f32_model = MyNet(numChannels=1, classes=10).to(cpu_device)
path = args["path"]
f32_model.load_state_dict(torch.load(path, map_location="cpu"))
f32_model_to_quantize = deepcopy(f32_model)
f32_model.eval()
f32_model_to_quantize.eval()
print(f32_model)

'''
MyNet(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (fc1): Linear(in_features=64, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=10, bias=True)
'''

modules_to_fuse = [
    ['conv1', 'bn1'],
    ['conv2', 'bn2'],
    ['conv3', 'bn3']
]

fused_model = torch.quantization.fuse_modules(f32_model_to_quantize, modules_to_fuse, inplace=True)
print(fused_model)


class quantStubModel(nn.Module):
    def __init__(self, model_fp32):
        super(quantStubModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()

        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()

        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


# creating nn.Module with stubs for inputs and outputs
quant_stubbed_model = quantStubModel(model_fp32=fused_model);

# Use FBGEMM for x86 Architecture & Use qnnpack for ARM Architecture
use_fbgemm = True

if args["arch"] == 'arm64':
    # Use arm64 device such as Raspi device.
    use_fbgemm = False

if use_fbgemm == True:
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')
    torch.backends.quantized.engine = 'fbgemm'

else:
    quantization_config = torch.quantization.default_qconfig
    torch.backends.quantized.engine = 'qnnpack'

"""
Prepare
"""
quant_stubbed_model.qconfig = quantization_config
torch.quantization.prepare(quant_stubbed_model, inplace=True)

quantSet = torch.utils.data.Subset(test_set, indices=np.arange(24))
quantDataloader = torch.utils.data.DataLoader(quantSet, batch_size=batch_size_test)

'''
Calibration 
'''
test_iter = iter(test_loader)
inputs, labels = next(test_iter)
with torch.no_grad():
    #    for inputs, labels in tqdm(quantDataloader):
    inputs, labels = inputs.to(cpu_device), labels.to(cpu_device)
    _ = quant_stubbed_model(inputs)
"""
Training Loop
"""
"""Training Loop"""
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(quant_stubbed_model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(cpu_device)
        labels = labels.to(cpu_device)

        # Forward pass
        outputs = quant_stubbed_model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

"""
Convert
"""
quant_stubbed_model.eval()
quantized_model = torch.quantization.convert(quant_stubbed_model, inplace=True)

#torch.jit.save(torch.jit.script(quantized_model), 'models/int8_qat_jit_model.pt')

test_ = iter(test_loader)
images, labels = next(test_)
print("shape {} ".format(images.shape))

print("[INFO] predicting...")

data = images[0]
label = labels[0]
print("data.shape {} ".format(data.shape))
data = torch.unsqueeze(data, dim=0)
print("after unsqueeze data.shape {} ".format(data.shape))
data.to("cpu")

torch.jit.save(torch.jit.trace(quantized_model, data), 'models/int8_qat_jit_model.pt')
int8_jit_model = torch.jit.load('models/int8_qat_jit_model.pt', map_location=cpu_device)

print(int8_jit_model)
int8_jit_model.eval()
test_ = iter(test_loader)
images, labels = next(test_)
print("shape {} ".format(images.shape))

print("[INFO] predicting...")

data = images[0]
label = labels[0]
print("data.shape {} ".format(data.shape))
data = torch.unsqueeze(data, dim=0)
print("after unsqueeze data.shape {} ".format(data.shape))
data.to("cpu")
int8_jit_model.to("cpu")
print("data.shape {} ".format(data.shape))

start = time.time()
output = int8_jit_model(data)
end = time.time()
print("Elapsed time = {} ms", (end - start) * 1000)
# max returns (value ,index)
_, predicted = torch.max(output.data, 1)

print("output.shape: {}", predicted.shape)
print("preds: {}", predicted)
print("label: {}", label)
