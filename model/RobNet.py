import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, convs=2):
    super().__init__()
    layers = [
      nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    ]
    for conv in range(convs-1):
      layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
      layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.ReLU(inplace=True))
    self.layers = nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.layers(x)
    return x

class RobNet(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.maxpool = nn.MaxPool2d(2, 2)
    self.block1 = BasicBlock(2, 64, 7, 3)
    self.block2 = BasicBlock(64, 128, 5, 2)
    self.block3 = BasicBlock(128, 256, 5, 2)
    self.block4 = BasicBlock(256, 512, 5, 2)
    self.block5 = BasicBlock(512, 1024, 5, 2)
    self.block6 = BasicBlock(1024, 2048, 3, 1)
    self.block7 = BasicBlock(2048, 4096, 3, 1)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(4096, 4096*4)
    self.fc2 = nn.Linear(4096*4, num_classes)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x): # [b, 2, 512, 2048]
    x = self.block1(x) # [b, 64, 512, 2048]
    x = self.maxpool(x) # [b, 64, 256, 1024]
    x = self.block2(x) # [b, 64, 256, 1024]
    x = self.maxpool(x) # [b, 128, 128, 512]
    x = self.block3(x) # [b, 256, 128, 512]
    x = self.maxpool(x) # [b, 256, 64, 256]
    x = self.block4(x) # [b, 512, 64, 256]
    x = self.maxpool(x) # [b, 512, 32, 128]
    x = self.block5(x) # [b, 1024, 32, 128]
    x = self.maxpool(x) # [b, 1024, 16, 64]
    x = self.block6(x) # [b, 2048, 16, 64]
    x = self.maxpool(x) # [b, 2048, 8, 32]
    x = self.block7(x) # [b, 4096, 8, 32]
    x = self.avgpool(x) # [b, 4096, 1, 1]
    x = torch.flatten(x, 1) # [b, 4096]
    x = self.fc1(x) # [b, 16384]
    x = self.fc2(x) # [b, 10]
    x = self.softmax(x) # [b, 10]
    return x

class RobNetLight(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.maxpool = nn.MaxPool2d(2, 2)
    self.block1 = BasicBlock(2, 16, 7, 3)
    self.block2 = BasicBlock(16, 32, 5, 2)
    self.block3 = BasicBlock(32, 64, 5, 2)
    self.block4 = BasicBlock(64, 128, 5, 2)
    self.block5 = BasicBlock(128, 256, 5, 2)
    self.block6 = BasicBlock(256, 512, 5, 1)
    self.block7 = BasicBlock(512, 1024, 5, 1)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(1024, 1024*4)
    self.fc2 = nn.Linear(1024*4, num_classes)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x): # [b, 2, 512, 2048]
    x = self.block1(x) # [b, 64, 512, 2048]
    x = self.maxpool(x) # [b, 64, 256, 1024]
    x = self.block2(x) # [b, 64, 256, 1024]
    x = self.maxpool(x) # [b, 128, 128, 512]
    x = self.block3(x) # [b, 256, 128, 512]
    x = self.maxpool(x) # [b, 256, 64, 256]
    x = self.block4(x) # [b, 512, 64, 256]
    x = self.maxpool(x) # [b, 512, 32, 128]
    x = self.block5(x) # [b, 1024, 32, 128]
    x = self.maxpool(x) # [b, 1024, 16, 64]
    x = self.block6(x) # [b, 2048, 16, 64]
    x = self.maxpool(x) # [b, 2048, 8, 32]
    x = self.block7(x) # [b, 4096, 8, 32]
    x = self.avgpool(x) # [b, 4096, 1, 1]
    x = torch.flatten(x, 1) # [b, 4096]
    x = self.fc1(x) # [b, 16384]
    x = self.fc2(x) # [b, 10]
    x = self.softmax(x) # [b, 10]
    return x

class RobNetLight2048(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.maxpool = nn.MaxPool2d(2, 2)
    self.block1 = BasicBlock(2, 16, 13, 6, 2)
    self.block2 = BasicBlock(16, 32, 11, 5, 2)
    self.block3 = BasicBlock(32, 64, 9, 4, 2)
    self.block4 = BasicBlock(64, 128, 7, 3, 3)
    self.block5 = BasicBlock(128, 256, 5, 2, 3)
    self.block6 = BasicBlock(256, 512, 5, 2, 3)
    self.block7 = BasicBlock(512, 1024, 3, 1, 3)
    self.block8 = BasicBlock(1024, 2048, 3, 1, 3)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(
      nn.Linear(2048, 2048*4),
      nn.ReLU(True),
      # nn.Dropout(.1),
      nn.Linear(2048*4, 2048*4),
      nn.ReLU(True),
      # nn.Dropout(.1),
      nn.Linear(2048*4, num_classes),
      nn.Softmax(dim=1)
    )
  
  def forward(self, x): # [b, 2, 1024, 1024]
    x = self.block1(x) # [b, 16, 1024, 1024]
    x = self.maxpool(x) # [b, 16, 512, 512]
    x = self.block2(x) # [b, 32, 512, 512]
    x = self.maxpool(x) # [b, 32, 256, 256]
    x = self.block3(x) # [b, 64, 256, 256]
    x = self.maxpool(x) # [b, 64, 128, 128]
    x = self.block4(x) # [b, 128, 128, 128]
    x = self.maxpool(x) # [b, 128, 64, 64]
    x = self.block5(x) # [b, 256, 64, 64]
    x = self.maxpool(x) # [b, 256, 32, 32]
    x = self.block6(x) # [b, 512, 32, 32]
    x = self.maxpool(x) # [b, 512, 16, 16]
    x = self.block7(x) # [b, 1024, 16, 16]
    x = self.maxpool(x) # [b, 1024, 8, 8]
    x = self.block8(x) # [b, 2048, 8, 8]
    x = self.avgpool(x) # [b, 2048, 1, 1]
    x = torch.flatten(x, 1) # [b, 2048]
    x = self.classifier(x)
    return x

class RobNetMel(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.maxpool = nn.MaxPool2d(2, 2)
    self.block1 = BasicBlock(1, 64, 13, 6, 2)
    self.block2 = BasicBlock(64, 128, 11, 5, 2)
    self.block3 = BasicBlock(128, 256, 9, 4, 2)
    self.block4 = BasicBlock(256, 512, 7, 3, 3)
    self.block5 = BasicBlock(512, 1024, 5, 2, 3)
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.classifier = nn.Sequential(
      nn.Linear(50176, 4096),
      nn.ReLU(True),
      # nn.Dropout(.1),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      # nn.Dropout(.1),
      nn.Linear(4096, num_classes),
      nn.Softmax(dim=1)
    )
  
  def forward(self, x): # [b, 2, 128, 320]
    x = self.block1(x) # [b, 64, 128, 320]
    x = self.maxpool(x) # [b, 64, 64, 160]
    x = self.block2(x) # [b, 128, 64, 160]
    x = self.maxpool(x) # [b, 128, 32, 80]
    x = self.block3(x) # [b, 256, 32, 80]
    x = self.maxpool(x) # [b, 256, 16, 40]
    x = self.block4(x) # [b, 512, 16, 40]
    x = self.maxpool(x) # [b, 512, 8, 20]
    x = self.block5(x) # [b, 1024, 8, 20]
    x = self.avgpool(x) # [b, 1024, 7, 7]
    x = torch.flatten(x, 1) # [b, 50176]
    x = self.classifier(x)
    return x

class RobNetMid(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    self.maxpool = nn.MaxPool2d(2, 2)
    self.block1 = BasicBlock(2, 16, 7, 3)
    self.block2 = BasicBlock(16, 32, 5, 2)
    self.block3 = BasicBlock(32, 64, 5, 2)
    self.block4 = BasicBlock(64, 128, 5, 2, 3)
    self.block5 = BasicBlock(128, 256, 5, 2, 3)
    self.block6 = BasicBlock(256, 512, 5, 2, 3)
    self.block7 = BasicBlock(512, 1024, 5, 2, 3)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(1024, 1024*4)
    self.fc2 = nn.Linear(1024*4, num_classes)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x): # [b, 2, 512, 2048]
    x = self.block1(x) # [b, 64, 512, 2048]
    x = self.maxpool(x) # [b, 64, 256, 1024]
    x = self.block2(x) # [b, 64, 256, 1024]
    x = self.maxpool(x) # [b, 128, 128, 512]
    x = self.block3(x) # [b, 256, 128, 512]
    x = self.maxpool(x) # [b, 256, 64, 256]
    x = self.block4(x) # [b, 512, 64, 256]
    x = self.maxpool(x) # [b, 512, 32, 128]
    x = self.block5(x) # [b, 1024, 32, 128]
    x = self.maxpool(x) # [b, 1024, 16, 64]
    x = self.block6(x) # [b, 2048, 16, 64]
    x = self.maxpool(x) # [b, 2048, 8, 32]
    x = self.block7(x) # [b, 4096, 8, 32]
    x = self.avgpool(x) # [b, 4096, 1, 1]
    x = torch.flatten(x, 1) # [b, 4096]
    x = self.fc1(x) # [b, 16384]
    x = self.fc2(x) # [b, 10]
    x = self.softmax(x) # [b, 10]
    return x