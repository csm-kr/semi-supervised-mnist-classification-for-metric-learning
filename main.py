import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import visdom
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as tfs
from model import EmbeddingNet
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import SEMI_MNIST
from loss import MetricCrossEntropy


def main():
    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume', type=int, default=13)
    opts = parser.parse_args()
    print(opts)

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom()

    # 4. dataset
    mean, std = 0.1307, 0.3081

    transform = tfs.Compose([tfs.Normalize((mean, ), (std, ))])
    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((mean,), (std,))])

    train_set = MNIST('./data/MNIST',
                      train=True,
                      download=True,
                      transform=None)

    train_set = SEMI_MNIST(train_set,
                           transform=transform)

    test_set = MNIST('./data/MNIST',
                     train=False,
                     download=True,
                     transform=test_transform)

    # 5. data loader
    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=opts.batch_size,
                              num_workers=4,
                              pin_memory=True
                              )

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=opts.batch_size,
                             )

    # 6. model
    model = EmbeddingNet().to(device)

    # 7. criterion
    criterion = MetricCrossEntropy().to(device)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=0.9,
                                weight_decay=5e-5)

    # optimizer = optim.Adam(params=model.parameters(),
    #                        lr=opts.lr,
    #                        weight_decay=5e-5)

    # 9. scheduler
    scheduler = StepLR(optimizer=optimizer,
                       step_size=10,
                       gamma=0.5)
    # 10. resume
    if opts.resume:
        model.load_state_dict(torch.load('./saves/state_dict.{}'.format(opts.resume)))
        print("resume from {} epoch..".format(opts.resume))
    else:
        print("no checkpoint to resume.. train from scratch.")
    # --
    for epoch in range(opts.resume + 1, opts.epoch):
        # 11. trian
        for idx, (imgs, targets, samples, is_known) in enumerate(train_loader):
            model.train()
            batch_size = opts.batch_size

            imgs = imgs.to(device)  # [N, 1, 28, 28]
            targets = targets.to(device)  # [N]
            samples = samples.to(device)  # [N, 1, 32, 32]
            is_known = is_known.to(device)

            samples = samples.view(batch_size * 10, 1, 28, 28)
            out_x = model(imgs)  # [N, 10]
            out_z = model(samples).view(batch_size, 10, out_x.size(-1))  # [N * 10 , 2]
            loss = criterion(out_x, targets, out_z, is_known, 1, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vis.line(X=torch.ones((1, 1)) * idx + epoch * len(train_loader),
                     Y=torch.Tensor([loss]).unsqueeze(0),
                     update='append',
                     win='loss',
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='loss',
                               legend=['total_loss']))
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            if idx % 10 == 0:
                print('Epoch : {}\t'
                      'step : [{}/{}]\t'
                      'loss : {}\t'
                      'lr   : {}\t'
                      .format(epoch,
                              idx, len(train_loader),
                              loss,
                              lr))

        torch.save(model.state_dict(), './saves/state_dict.{}'.format(epoch))

        # 12. test
        correct = 0
        avg_loss = 0
        for idx, (img, target) in enumerate(test_loader):

            model.load_state_dict(torch.load('./saves/state_dict.{}'.format(epoch)))
            model.eval()
            img = img.to(device)         # [N, 1, 28, 28]
            target = target.to(device)   # [N]
            output = model(img)          # [N, 10]

            # output = torch.softmax(output, -1)
            pred, idx_ = output.max(-1)
            print(idx_)
            correct += torch.eq(target, idx_).sum()
            #loss = criterion(output, target)
            #avg_loss += loss.item()

        print('Epoch {} test : '.format(epoch))
        accuracy = correct.item() / len(test_set)
        print("accuracy : {:.4f}%".format(accuracy * 100.))
        #avg_loss = avg_loss / len(test_loader)
        #print("avg_loss : {:.4f}".format(avg_loss))

        vis.line(X=torch.ones((1, 1)) * epoch,
                 Y=torch.Tensor([accuracy]).unsqueeze(0),
                 update='append',
                 win='test',
                 opts=dict(x_label='epoch',
                           y_label='test_',
                           title='test_loss',
                           legend=['accuracy']))
        scheduler.step()


if __name__ == '__main__':
    main()
