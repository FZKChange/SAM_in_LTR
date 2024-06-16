import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.nn import Parameter
from torchvision import transforms
from general import IMBALANCECIFAR100, IMBALANCECIFAR10
import time
from losses import LDAMLoss
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__all__ = ['resnet32']

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 250:
        lr = lr * 0.0001
    elif epoch > 150:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels,in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x,conv_weights):
        batch_size, channels, height, width = x.size()
        batch_size, channels, height, width = x.size()
        # print(x.size())
        # print(conv_weights.size())
        conv_batch_size = conv_weights.size(0)
        conv_weights = F.avg_pool2d(conv_weights, kernel_size=7, stride=7)
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

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Normalizer():
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(self, model):  # this method applies tau-normalization on the classifier layer

        for curLayer in [model.linear.weight]:  # change to last layer: Done
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



# class NormedLinear(nn.Module):
#
#     def __init__(self, in_features, out_features):
#         super(NormedLinear, self).__init__()
#         self.weight = Parameter(torch.Tensor(in_features, out_features))
#         self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#
#     def forward(self, x):
#         out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
#         return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # self.shortcut = nn.Sequential(
            #          nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #          nn.BatchNorm2d(self.expansion * planes)
            #     )
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, conv_weights = None):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        self.conv_weights = conv_weights
        self.self_attention = SelfAttention(64 * block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.self_attention(out,self.conv_weights)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
feature_maps = torch.load('Feature_Maps_Resnet32_for_cifar10_r10_add_LADM_original_350.pt')
model = ResNet_s(BasicBlock, [5, 5, 5], num_classes=10, use_norm=False, conv_weights=feature_maps[0].to(device))
my_normalizer = Normalizer(LpNorm=2, tau=1.99)
my_normalizer.apply_on(model)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = IMBALANCECIFAR10(root='./data', train=True, download=False, transform=transform_train)
val_dataset = IMBALANCECIFAR10(root='./data',train=False,download=False, transform=transform_val)




train_sampler = None

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=2)





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
        print_freq =100

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
            print_freq = 100

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





optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9,
                            weight_decay=2e-4)

model.to(device)
feature_maps_list = []
if __name__ == '__main__':
    best_acc1 = 0
    start_epoch = 0
    epochs = 300
    lr = 0.01

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_sampler = None
        cls_num_list =  [5000, 3871, 2997, 2320, 1796, 1391, 1077, 834, 645, 500]
        idx = epoch // 150
        betas = [0, 0.999, 0.999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(device)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1= validate(val_loader, model, criterion, epoch, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)

        # if (epoch + 1) % 1 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         for data, _ in val_loader:
        #             data = data.to(device)
        #             out = F.relu(model.bn1(model.conv1(data)))
        #             out = model.layer1(out)
        #             out = model.layer2(out)
        #             out = model.layer3(out)
        #             out = model.self_attention(out)
        #             features = out  # 这里使用conv1替代layer4[-1].conv2
        #             break  # 只提取第一个批次的特征图
        #
        #     feature_maps_list.append(features.detach().cpu())
        #
        #     torch.save(feature_maps_list, f'Feature_Maps_Resnet32_for_cifar10_r100_add_LADM_original_{epoch + 1}.pt')
        #     print(f'Feature maps saved for ResNet32 epoch {epoch + 1}')



