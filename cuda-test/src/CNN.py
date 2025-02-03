# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bVY_QQ0NCJGBDLT3_YNQSAsSbZbkwVyG
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random

import kagglehub

path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Path to dataset files:", path)

train_dogs = []
train_cats = []
test_dogs = []
test_cats = []
# 0 is dog, 1 is cat

def load_all(folder_path, index_set):
  imgs = []
  for i, filename in enumerate(os.listdir(folder_path)):
    if i in index_set:
      file_path = os.path.join(folder_path, filename)
      img = Image.open(file_path)
      imgs.append(img)
  return imgs

dog_path = os.path.join(path, 'PetImages', 'Dog')
cat_path = os.path.join(path, 'PetImages', 'Cat')
num_dogs = len(os.listdir(dog_path))
num_cats = len(os.listdir(cat_path))
train_size = 5000
test_size = 100

test_index_set_dog = set(random.sample(range(train_size + test_size), test_size))
test_index_set_cat = set(random.sample(range(train_size + test_size), test_size))
train_index_set_dog = set(range(train_size + test_size)) - test_index_set_dog
train_index_set_cat = set(range(train_size + test_size)) - test_index_set_cat

print(f"dog images: {num_dogs}   cat images: {num_cats}")
print(f"Training size: {train_size} cats, {train_size} dogs")
print(f"Test dataset size: {test_size} cats, {test_size} dogs")
train_dogs = load_all(dog_path, train_index_set_dog)
train_cats = load_all(cat_path, train_index_set_cat)
test_dogs = load_all(dog_path, test_index_set_dog)
test_cats = load_all(cat_path, test_index_set_cat)

plt.imshow(test_cats[1].convert('L').resize((128, 128)))
arr = np.array(test_cats[1])
print(arr[0][0])

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class data(Dataset):
  def __init__(self, cat_images, dog_images, size=(64, 64)):
    #scrambles them and puts them into a torch tensor
    cat_rem = len(cat_images)
    dog_rem = len(dog_images)
    images = []
    ans = []
    while cat_rem + dog_rem > 0:
      rand = random.random()
      if rand <= cat_rem / (cat_rem + dog_rem):
        #choose cat image
        img = cat_images[len(cat_images) - cat_rem]
        ans.append(1)
        cat_rem -= 1
      else:
        #choose dog image
        img = dog_images[len(dog_images) - dog_rem]
        dog_rem -= 1
        ans.append(0)
      img = img.convert('L').resize(size)
      img = np.array(img, dtype=np.float32).tolist()
      images.append(img)
    self.train_imgs = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    self.train_ans = torch.tensor(ans, dtype=torch.long)
  def __len__(self):
    return len(self.train_ans)
  def __getitem__(self, index):
    return self.train_imgs[index], self.train_ans[index]

train_data = data(train_cats, train_dogs)
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = data(test_cats, test_dogs)
batch_size = 32
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class CNNBinaryClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(128 * 6 * 6, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 2)
  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = self.pool(self.relu(self.conv3(x)))

    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

model = CNNBinaryClassifier()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

model = CNNBinaryClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def train(model, train_dataloader, loss_fn, optimizer):
  model.train()
  train_loss = 0
  for batch, (img, ans) in enumerate(train_dataloader):
    pred_out = model(img)
    loss = loss_fn(pred_out, ans)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"Total loss: {train_loss}")

def test(model, test_dataloader):
  global disp_imgs
  model.eval()
  dog_right = 0
  dog_total = 0
  cat_right = 0
  cat_total = 0
  for batch, (img, ans) in enumerate(test_dataloader):
    with torch.no_grad():
      pred = model(img)
    for i in range(pred.size(0)):
      out_index = torch.argmax(pred[i])
      out_ans = ans[i].item()
      if out_ans == 0: # dog
        dog_total += 1
        if out_index == out_ans:
          dog_right += 1
      if out_ans == 1: # cat
        cat_total += 1
        if out_index == out_ans:
          cat_right += 1
  return dog_right, dog_total, cat_right, cat_total

epochs = 100
for i in range(epochs):
  train(model, train_dataloader, loss_fn, optimizer)
  dog_right, dog_total, cat_right, cat_total = test(model, test_dataloader)
  print(f"Epoch {i}: {dog_right}/{dog_total} dog images correct, {cat_right}/{cat_total} cat images correct. {(dog_right + cat_right) / (dog_total + cat_total)} accuracy")
