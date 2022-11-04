# pt_dnn.py --type load/train --jit-model-name 32_jit_dnn.pt --model-name 32_dnn.pth --path models --loadjit 1
import torch
import torchvision
import os
import torch.nn as nn
import argparse
import time
from simple_network import MyNet

from torch.optim import lr_scheduler

ap = argparse.ArgumentParser()
ap.add_argument('--type', default='load', choices=['load', 'train'], help='Use train for training else use load for loading & testing')

ap.add_argument("-j", "--jitmname", required=True,
	help="name of jit model to be saved")
ap.add_argument("-n", "--mname", required=True,
	help="name of the model to be saved")
ap.add_argument("-p", "--path", required=True,
	help="path where model can be saved")
ap.add_argument("-u", "--loadjit", type=int, default=1,
	help="Use 1 for loading jit model else use 0 for loading normal model")

args = vars(ap.parse_args())

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

batch_size_train = 64
batch_size_test = 64

random_seed = 1
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.RandomRotation(20),
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



def train():
    model = MyNet(numChannels=1, classes=10).to(device)
    print(model)

    # Hyper-parameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    n_total_steps = len(train_loader)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    #linear_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=n_total_steps, epochs=10, anneal_strategy='linear')
    # Train the model
    for epoch in range(num_epochs):
        #exp_lr_scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            criterion.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        exp_lr_scheduler.step()
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')

    jit_model_name = os.path.join(args['path'], args['jitmname'])
    model_name = os.path.join(args['path'], args['mname'])

    print('jit_model_name {}, model_name {}'.format(jit_model_name, model_name))
    #model_scripted = torch.jit.script(model)
    test_ = iter(test_loader)
    images, labels = next(test_)
    data = images[0]
    label = labels[0]
    data = torch.unsqueeze(data, dim=0)
    data.to('cpu')
    label.to('cpu')
    example = torch.rand(1, 1, 28, 28)
    example.to(device)
    model.to('cpu')

    model_scripted = torch.jit.trace(model, data)
    torch.save(model.state_dict(), model_name)
    #torch.save(model_scripted.state_dict(), jit_model_name)
    torch.jit.save(model_scripted, jit_model_name)

def load() :
    jit_script = args["loadjit"]
    model = MyNet(numChannels=1, classes=10).to(device)
    path = args["path"]
    jit_model_name = os.path.join(args['path'], args['jitmname'])
    model_name = os.path.join(args['path'], args['mname'])
    if jit_script:
        model = torch.jit.load(jit_model_name, map_location="cpu")
    else:
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
    print(model)
    model.eval()
    test_ = iter(test_loader)
    images, labels = next(test_)
    print("shape {} ".format(images.shape))

    print("[INFO] predicting...")
    if jit_script:
        data = images[0]
        label = labels[0]
        data = torch.unsqueeze(data, dim=0)
    else:
        data = images[0:64]
        label = labels[0:64]
    print("data.shape {} ".format(data.shape))

    print("after unsqueeze data.shape {} ".format(data.shape))
    data.to("cpu")
    model.to("cpu")
    print("data.shape {} ".format(data.shape))
    start = time.time()
    output = model(data)
    end = time.time()
    output.to("cpu")
    print("Elapsed time = {} ms", (end - start) * 1000)
    # max returns (value ,index)
    _, predicted = torch.max(output.data, 1)

    print("output.shape: {}", predicted.shape)
    print("preds: {}", predicted)
    print("label: {}", label)
    # loop over the sample images

if args["type"] == 'train':
    train()
if args["type"] == 'load':
    load()
'''
[CNN Architecture Used for quantization]
model.add(lq.layers.QuantConv2D(32, (3, 3),
                                kernel_quantizer=None,
                                kernel_constraint=None,
                                use_bias=False,
                                input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(64, use_bias=False))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantDense(10, use_bias=False))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))
model = ConvNet().to(device)
'''