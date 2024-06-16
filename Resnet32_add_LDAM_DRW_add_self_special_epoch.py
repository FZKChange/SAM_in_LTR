import torch
import torch.nn as nn
import pickle
import math
from losses import LDAMLoss
import time
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from general import IMBALANCECIFAR10,IMBALANCECIFAR100
from utils import *
def adjust_batch_size(dataloader, current_batch_size):
    if len(dataloader.dataset) < current_batch_size:
        new_batch_size = len(dataloader.dataset)
    else:
        new_batch_size = current_batch_size
    dataloader.batch_size = new_batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels,in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, conv_weights):
        batch_size, channels, height, width = x.size()
        conv_batch_size = conv_weights.size(0)

        # If conv_weights' batch size is not equal to the current batch size, adjust query size
        if conv_batch_size != batch_size:
            query = self.query(conv_weights[:batch_size]).view(batch_size, -1, height * width).permute(0, 2, 1)
        else:
            query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)
        # query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)
        # if conv_weights == 0:
        #     query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # else:
        #     conv_batch_size = conv_weights.size(0)
        #
        #     if conv_batch_size != batch_size:
        #         query = self.query(conv_weights[:batch_size]).view(batch_size, -1, height * width).permute(0, 2, 1)
        #     else:
        #         query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)

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

class ResNet32(nn.Module):
    def __init__(self, block, layers, num_classes = 100, conv_weights = None):
        super(ResNet32, self).__init__()
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
        self.conv_weights = conv_weights
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
        x = self.self_attention(x,self.conv_weights)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def validate(val_loader, model, criterion, epoch, device, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # tf_writer = SummaryWriter(log_dir='./log/pure_Resnet32/directory')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            print_freq = 4

            if i % print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()
        #
        # tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg

train_transform = transforms.Compose([
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
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.__dict__[args.arch](num_classes=365, phase_train=False)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model
# 定义特定类别的标签
trainset = IMBALANCECIFAR100(root='./data', train=True, download=False, transform=train_transform)
testset = IMBALANCECIFAR100(root='./data',train=False,download=False, transform=test_transform)
# train_dataset = imbalanced_train_datasetok


# test_dataset = imbalanced_test_dataset
#

# 创建 DataLoader 对象用于批量加载训练集和测试集数据
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True,num_workers=4)

test_dataloader = DataLoader(testset, batch_size=64, shuffle=False,num_workers=4)
feature_maps = torch.load('Feature_Maps_Resnet32_for_cifar100_r100_add_LADM_250.pt')
#
# dam_loss = LDAMLoss(cls_num_list, max_m=0.5, weight=None)  # 根据需要调整max_m


model = ResNet32(ResNetBlock, [5, 5, 5],num_classes=100,conv_weights=feature_maps[0].to(device))
checkpoint = torch.load('checkpoints/Resnet32_add_LDAM_DRW_idx_250_checkpoint_epoch_100.pth')  # 指定需要加载的 checkpoint 文件
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
num_epochs = 110
best_top1 = 0
best_top5 = 0
# 训练模型

if __name__ == '__main__':
    # 训练模型
    print('2023.10.12.9:31现在训练的是cifar100_r100的idx为250的self_LDAM_DRW')
    for epoch in range(num_epochs):
        cls_num_list = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]

        idx = epoch // 60


        betas = [0, 0.999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(device)

        for i, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)  # 将输入数据移动到 GPU 上
            optimizer.zero_grad()
            output = model(data)
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印训练日志
            if i % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}')
            # 在每个epoch结束后进行验证

        top1_accuracy = validate(val_loader=test_dataloader, model=model, criterion=criterion, epoch=epoch,device=device,
                                 flag='val')


        # 更新best@1和best@5
        if top1_accuracy > best_top1:
            best_top1 = top1_accuracy





            # 更新学习率
        scheduler.step()

        print('Training complete.')
        print(f'Best Top1 Accuracy: {best_top1}')
        # if (epoch + 1) % 100 == 0:
        #         checkpoint = {
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'best_top1': best_top1
        #         }
        #         checkpoint_path = os.path.join('checkpoints', f'Resnet32_add_LDAM_DRW_idx_250_checkpoint_epoch_{epoch + 1}.pth')
        #         os.makedirs('checkpoints', exist_ok=True)
        #         torch.save(checkpoint, checkpoint_path)
        #         print(f'Checkpoint saved at {checkpoint_path}')


