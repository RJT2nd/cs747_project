from preprocessing.gtzan_dataset import GTZAN_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from model.RobNet import RobNet
import math
from torch.utils.data import random_split
from tqdm import tqdm

def getDataLoaders(batch_size):
  dataset = GTZAN_Dataset(load_to_memory=False, mel=True)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  return train_dataloader, test_dataloader

def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
  model.train()
  loss_sum = 0
  num_correct = 0
  total_samples = 0
  for i, (images, labels) in enumerate(tqdm(dataloader)):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_sum += loss.item()
    print(outputs)
    print(loss_sum)

    num_correct += torch.sum(torch.argmax(labels, dim=1) == predictions)
    total_samples += len(images)

  f_loss = math.floor(loss_sum / total_samples * 1000) / 1000
  f_accuracy = math.floor(num_correct / total_samples * 1000) / 1000

  print(f'Training Loss[{f_loss}] Acc[{f_accuracy}] Epoch {epoch+1}')
  return f_loss, f_accuracy

def validate_epoch(model, dataloader, criterion, optimizer, epoch, device):
  model.eval()
  loss_sum = 0
  num_correct = 0
  total_samples = 0
  for i, (images, labels) in enumerate(tqdm(dataloader)):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)

    loss = criterion(outputs, labels)

    loss_sum += loss.item()

    num_correct += torch.sum(torch.argmax(labels, dim=1) == predictions)
    total_samples += len(images)

  f_loss = math.floor(loss_sum / total_samples * 1000) / 1000
  f_accuracy = math.floor(num_correct / total_samples * 1000) / 1000

  print(f'Validation Loss[{f_loss}] Acc[{f_accuracy}] Epoch {epoch+1}')
  return f_loss, f_accuracy

def train(model, epochs, batch_size, learning_rate):
  train_dataloader, test_dataloader = getDataLoaders(batch_size)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  best_acc = 0
  for epoch in range(epochs):
    print(f'\nEpoch {epoch+1}')

    train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, epoch, device)
    test_loss, test_acc = validate_epoch(model, train_dataloader, criterion, optimizer, epoch, device)

    if test_acc > best_acc:
      print('Saving checkpoint...')
      best_acc = test_acc
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_acc,
      }, 'checkpoints/chechpoint.pt')
