'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model import *
from model.utils import progress_bar

#
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--trainbs', default=128, type=int, help='trainloader batch size')
# parser.add_argument('--testbs', default=100, type=int, help='testloader batch size')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#
# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.trainbs, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.testbs, shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# # Model
# print('==> Building model..')
# net = VGG('VGG16')
# # net = ResNet18()
# # net = PreActResNet18()
# # net = GoogLeNet()
# # net = DenseNet121()
# # net = ResNeXt29_2x64d()
# # net = ResNeXt29_32x4d()
# # net = MobileNet()
# # net = MobileNetV2()
# # net = DPN92()
# # net = ShuffleNetG2()
# # net = SENet18()
# # net = ShuffleNetV2(1)
# # net = EfficientNetB0()
# net_name = net.name
# save_path = './checkpoint/{0}_ckpt.pth'.format(net.name)
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
#
# if args.resume:
#     # Load best checkpoint trained last time.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(save_path)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

# Training
best_acc=0
def train(epoch,model,optimizer,device,trainloader,criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss /= len(trainloader)
    print('train_loss',train_loss)
def test(epoch,model,testloader,device,criterion):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving ' + net_name + ' ..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, save_path)
        best_acc = acc

# if __name__ == '__main__':
#     for epoch in range(start_epoch, start_epoch+300):
#         # In PyTorch 1.1.0 and later,
#         # you should call them in the opposite order:
#         # `optimizer.step()` before `lr_scheduler.step()`
#         train(epoch)
#         test(epoch)
#         scheduler.step()  # 每隔100 steps学习率乘以0.1
#         lr = optimizer.param_groups[0]['lr']
#         print('lr ', lr)
#     print("\nTesting best accuracy:", best_acc)
