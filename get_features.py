#This is the code for getting the features from the tail classes
import torch
import torch.nn as nn
import pickle
import math

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from general import IMBALANCECIFAR10

def adjust_batch_size(dataloader, current_batch_size):
    if len(dataloader.dataset) < current_batch_size:
        new_batch_size = len(dataloader.dataset)
    else:
        new_batch_size = current_batch_size
    dataloader.batch_size = new_batch_size
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels,in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet32_add_self(nn.Module):
    def __init__(self, block, layers, num_classes=34):
        super(ResNet32_add_self, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.self_attention = SelfAttention(64 * block.expansion)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        x = self.self_attention(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
# 加载 new_train_dataset
with open('new_train_dataset.pkl', 'rb') as f:
    new_train_dataset = pickle.load(f)

# 加载 new_test_dataset
with open('new_test_dataset.pkl', 'rb') as f:
    new_test_dataset = pickle.load(f)

# 创建 DataLoader 对象用于批量加载训练集和测试集数据


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet32_add_self(ResNetBlock, [5, 5, 5])

# 创建 DataLoader 对象用于批量加载训练集和测试集数据
train_dataloader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(new_test_dataset, batch_size=64, shuffle=False)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 240

# 训练模型

feature_maps_list=[]

# 训练模型
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}')

    if (epoch + 1) % 10 == 0:
        model.eval()
        num_classes = 4
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            class_correct = [0] * num_classes
            class_total = [0] * num_classes
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to('cuda'), target.to('cuda')
                output = model(data)
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)

                for i in range(num_classes):
                    class_correct[i] += predicted[target == i].eq(i).sum().item()
                    class_total[i] += target[target == i].size(0)

            overall_accuracy = total_correct / total_samples

            print(f'Epoch [{epoch + 1}/{num_epochs}], Overall Accuracy: {overall_accuracy:.4f}')
            for i in range(num_classes):
                class_accuracy = class_correct[i] / class_total[i]
                print(f'Class {i} Accuracy: {class_accuracy:.4f}')
            print('---')

        scheduler.step()

    print('Training complete.')
    if (epoch + 1) % 40 == 0:
        model.eval()
        with torch.no_grad():
            for data, _ in train_dataloader:
                data = data.to(device)
                out = model.conv1(data)
                out = model.bn1(out)
                out = model.relu(out)
                out = model.maxpool(out)

                out = model.layer1(out)
                out = model.layer2(out)
                out = model.layer3(out)
                out = model.self_attention(out)
                features = out  # 这里使用conv1替代layer4[-1].conv2
                break  # 只提取第一个批次的特征图

        feature_maps_list.append(features.detach().cpu())

        torch.save(feature_maps_list, f'Feature_Maps_Resnet32_for_10_{epoch + 1}.pt')
        print(f'Feature maps saved for ResNet32 epoch {epoch + 1}')
