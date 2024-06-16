import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torchvision import transforms
from places_data import Tiny_data
import time
from losses import LDAMLoss
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义了ResNetBlock

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
#         key = self.key(x).view(batch_size, -1, height * width)
#         energy = torch.bmm(query, key)
#         attention = torch.softmax(energy, dim=-1)
#
#         value = self.value(x).view(batch_size, -1, height * width)
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, channels, height, width)
#         out = self.gamma * out + x
#
#         return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, conv_weights):
        # batch_size, channels, height, width = x.size()
        # query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)
        # key = self.key(x).view(batch_size, -1, height * width)
        # energy = torch.bmm(query, key)
        # attention = torch.softmax(energy, dim=-1)
        #
        # value = self.value(x).view(batch_size, -1, height * width)
        # out = torch.bmm(value, attention.permute(0, 2, 1))
        # out = out.view(batch_size, channels, height, width)
        # out = self.gamma * out + x
        #
        # return out

        batch_size, channels, height, width = x.size()
        # print(x.size())
        # print(conv_weights.size())
        conv_batch_size = conv_weights.size(0)
        # conv_weights = F.avg_pool2d(conv_weights, kernel_size=7, stride=7)
        # Reshape conv_weights

        # query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)

        # If conv_weights' batch size is not equal to the current batch size, adjust query size
        if conv_batch_size != batch_size:
            query = self.query(conv_weights[:batch_size]).view(batch_size, -1, height * width).permute(0, 2, 1)
        else:
            query = self.query(conv_weights).view(batch_size, -1, height * width).permute(0, 2, 1)

        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)

        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


class ResNetBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=200,conv_weights=None, conv_bias=None):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = SelfAttention(512 * block.expansion)
        self.conv_weights = conv_weights
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x,self.conv_weights)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



# def adjust_learning_rate(optimizer, epoch, lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if epoch > 94:
#         lr = lr * 0.001
#
#     elif epoch > 79:
#         lr = lr * 0.01
#     elif epoch > 59:
#         lr = lr * 0.1
#     else:
#         if epoch <= 5:
#             lr = lr * epoch/5
#         else:
#            lr = lr
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Normalizer():
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(self, model):  # this method applies tau-normalization on the classifier layer

        for curLayer in [model.fc.weight]:  # change to last layer: Done
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (
                        torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)

            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape) - 1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset = Tiny_data('./data', transform_train, 'train')
val_dataset = Tiny_data('./data', transform_val, 'val')
train_sampler = None
feature_maps = torch.load('Feature_Maps_Resnet50_for_100_300.pt')
model = ResNet50(ResNetBlock, [3, 4, 6, 3],conv_weights=feature_maps[0].to(device))
# my_normalizer = Normalizer(LpNorm=2, tau=1.99)
# my_normalizer.apply_on(model)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=(train_sampler is None),
    num_workers=4, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print_freq =200

        if i % print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)


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
            print_freq = 200

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
    return top1.avg





optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=2e-4)

model.to(device)
feature_maps_list=[]
if __name__ == '__main__':
    best_acc1 = 0
    start_epoch = 0
    epochs = 300
    lr = 0.001

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_sampler = None
        cls_num_list = [500, 488, 477, 466, 455, 445, 435, 425, 415, 405, 396, 387, 378, 370, 361, 353, 345, 337, 329, 322, 314, 307, 300, 293, 286, 280, 273, 267, 261, 255, 249, 244, 238, 232, 227, 222, 217, 212, 207, 202, 198, 193, 189, 184, 180, 176, 172, 168, 164, 160, 157, 153, 150, 146, 143, 140, 136, 133, 130, 127, 124, 121, 119, 116, 113, 111, 108, 106, 103, 101, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 73, 71, 69, 68, 66, 65, 63, 62, 60, 59, 58, 56, 55, 54, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 34, 33, 32, 31, 31, 30, 29, 29, 28, 27, 27, 26, 25, 25, 24, 24, 23, 23, 22, 21, 21, 20, 20, 20, 19, 19, 18, 18, 17, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5]

        idx = epoch // 160
        betas = [0, 0.999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0-betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.2, s=50, weight=per_cls_weights).cuda(device)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        # if (epoch + 1) % 100 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         for data, _ in val_loader:
        #             data = data.to(device)
        #             #
        #             #
        #             # out = model.conv1(data)
        #             # out = model.bn1(out)
        #             # out = model.relu(out)
        #             # out = model.maxpool(out)
        #             #
        #             # out = model.layer1(out)
        #             # out = model.layer2(out)
        #             # out = model.layer3(out)
        #             # out = model.layer4(out)
        #
        #             x = model.conv1(data)
        #             x = model.bn1(x)
        #             x = model.relu(x)
        #             x = model.maxpool(x)
        #
        #             x = model.layer1(x)
        #             x = model.layer2(x)
        #             x = model.layer3(x)
        #             x = model.layer4(x)
        #             x = model.attention(x)
        #             features = x  # 这里使用conv1替代layer4[-1].conv2
        #             break  # 只提取第一个批次的特征图
        #
        #     feature_maps_list.append(features.detach().cpu())
        #
        #     torch.save(feature_maps_list, f'Feature_Maps_Resnet50_for_20_{epoch + 1}.pt')
        #     print(f'Feature maps saved for ResNet50 epoch {epoch + 1}')



