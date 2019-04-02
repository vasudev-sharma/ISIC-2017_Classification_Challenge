

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch import optim

from torchvision import datasets, transforms, models
from collections import OrderedDict


#Explore Data

import glob 
import cv2

train_dir = ["./data1/train/Melanoma", 
               "./data1/train/Nevus",
               "./data1/train/Seborrheic_keratosis"]

test_dir = ["./data1/test/Melanoma", 
              "./data1/test/Nevus",
              "./data1/test/Seborrheic_keratosis"]

def get_images_paths(directory, n_per_class):
    img_paths = []
    for i in range(3):
        cont = 1
        for file in glob.glob(directory[i]+"/*.jpg"):
            img_paths.append(file)
            if cont == n_per_class:
                break
            cont +=1 
    return img_paths

# Get a sample of 15 training images (5 for each class)
img_paths = get_images_paths(train_dir, 5)
#print(len(img_paths))
#print(img_paths)


# Read images 
images = []
for path in img_paths: 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

# Check if images has the same shape
for i in range(len(images)):
    print("image: ", i, " has shape: ", images[i].shape)

# Visualize images
index = 4

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title(img_paths[index])
#ax1.imshow(images[index])


#Check Model Architecture for transfer Learning

model = models.inception_v3(pretrained=True)
model


#Load Data

root_path = "./data1/"

# With data augmentation
train_transforms = transforms.Compose([transforms.Resize((350,350)),
                                       transforms.CenterCrop(299),
                                       transforms.RandomRotation(0, 359),
                                       transforms.RandomVerticalFlip(0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((300,300)),
                                       transforms.CenterCrop(299),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = valid_transforms

train_data = datasets.ImageFolder(root_path+'train', transform=train_transforms)
valid_data = datasets.ImageFolder(root_path+'valid', transform=valid_transforms)
test_data = datasets.ImageFolder(root_path+'test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

#Visualize different training images
import numpy as np 

def imshow(image, ax=None, title=None, normalize=True):

    if ax is None:
        fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

images, labels = next(iter(trainloader))
#imshow(images[7])


#Build Model

#Freeze Parameters
for parameters in model.parameters():
    parameters.requires_grad = False 

#Create Classifier

BasicConv2d = nn.Sequential(OrderedDict([('conv', nn.Conv2d(2048, 192, kernel_size=(1,1), stride=(1,1))),
                                         ('bn', nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))]))

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 512)),
                                         ('relu', nn.ReLU()),
                                         ('drop', nn.Dropout(0.5)),
                                         ('fc2', nn.Linear(512, 3)), 
                                         ('output', nn.Softmax(dim=1))]))

model.Mixed_7c.branch_pool = BasicConv2d
model.fc = classifier


#Train Classifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#Accuracy Before Training

def accuracy(validloader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            outputs, aux = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network %d %%' % (100 * correct / total))

model.to(device)
accuracy(validloader)


#Time on a single Batch

import time

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.fc.parameters(), lr=0.001)

model.to(device)
model.train()

inputs, labels = next(iter(trainloader))
inputs, labels = inputs.to(device), labels.to(device)

start = time.time()

outputs = model.forward(inputs)
# Inception has multiple outputs
loss = sum((criterion(out, labels) for out in outputs))
loss.backward()
optimizer.step()

print(f"Device = GPU; Time per batch: {(time.time() - start):.3f} seconds")

#Train Classifier

import time
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item()
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy
  
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, momentum=0, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

model.to(device)

running_loss = 0
step = 0
print_every = 1
epochs = 5


model.train()
start=time.time()
for epoch in range(epochs):
    scheduler.step()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        step += 1
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = sum((criterion(out, labels) for out in outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % print_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                print("Epoch: {}/{}.. ".format(epoch+1,epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)))
            running_loss = 0
            model.train()
print(f"Device = GPU; Time time for Training is : {(time.time() - start):.3f} seconds")
    



