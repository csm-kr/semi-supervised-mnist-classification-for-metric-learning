import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import visdom
from torchvision.datasets import MNIST
import torchvision.transforms as tfs
from model import EmbeddingNet
from torch.optim.lr_scheduler import StepLR
from dataset import SEMI_MNIST
from loss import MetricCrossEntropy
import numpy as np


def main():

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
                           transform=transform,
                           num_samples=100)

    test_set = MNIST('./data/MNIST',
                     train=False,
                     download=True,
                     transform=test_transform)

    test_set = SEMI_MNIST(test_set,
                          transform=transform,
                          num_samples=100)

    # 5. data loader
    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=1,
                              num_workers=8,
                              pin_memory=True
                              )

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=1,
                             )

    # 6. model
    model = EmbeddingNet().cuda()
    model.load_state_dict(torch.load('./saves/state_dict.{}'.format(15)))

    # 7. criterion
    criterion = MetricCrossEntropy()

    data = []
    y = []
    is_known_ = []
    # for idx, (imgs, targets, samples, is_known) in enumerate(train_loader):
    #     model.train()
    #     batch_size = 1
    #     imgs = imgs.cuda()  # [N, 1, 28, 28]
    #     targets = targets.cuda()  # [N]
    #     samples = samples.cuda() # [N, 1, 32, 32]
    #     is_known = is_known.cuda()
    #
    #     output = model(imgs)
    #     y.append(targets.cpu().detach().numpy())
    #     is_known_.append(is_known.cpu().detach().numpy())
    #
    #     if idx % 100 == 0:
    #         print(idx)
    #         print(output.size())
    #
    #     data.append(output.cpu().detach().numpy())
    #
    # data_numpy = np.array(data)
    # y_numpy = np.array(y)
    # is_known_numpy = np.array(is_known_)
    #
    # np.save('data', data_numpy)
    # np.save('known', is_known_numpy)
    # np.save('y', y)

    data_numpy = np.load('data.npy')
    y_numpy = np.load('y.npy')
    is_known_numpy = np.load('known.npy')

    print(data_numpy.shape)
    print(y_numpy.shape)

    data_numpy = np.squeeze(data_numpy)
    y_numpy = np.squeeze(y_numpy)
    is_known_numpy = np.squeeze(is_known_numpy)

    print(data_numpy.shape)
    print(y_numpy.shape)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
              '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf', '#ada699']

    # t-SNE 모델 생성 및 학습
    # tsne = TSNE(random_state=0)
    # digits_tsne = tsne.fit_transform(data_numpy)
    # np.save('tsne', digits_tsne)
    digits_tsne = np.load('tsne.npy')
    print('complete t-sne')

    # ------------------------------ 1 ------------------------------
    plt.figure(figsize=(10, 10))
    for i in range(11):
        inds = np.where(y_numpy == i)[0]
        known = is_known_numpy[inds]
        known_idx = np.where(known == 1)
        unknown_idx = np.where(known == 0)

        plt.scatter(digits_tsne[inds[unknown_idx], 0], digits_tsne[inds[unknown_idx], 1], alpha=0.5, color=colors[10])
        plt.scatter(digits_tsne[inds[known_idx], 0], digits_tsne[inds[known_idx], 1], alpha=0.5, color=colors[i])

        # plt.scatter(digits_tsne[inds, 0], digits_tsne[inds, 1], alpha=0.5, color=colors[i])

    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown'])
    plt.show()  # 그래프 출력

    # ------------------------------ 2 ------------------------------
    # # 시각화
    # for i in range(len(data_numpy)):  # 0부터  digits.data까지 정수
    #
    #     # plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(y_numpy[i]),  # x, y , 그룹
    #     #          color=colors[y_numpy[i]],  # 색상
    #     #          fontdict={'weight': 'bold', 'size': 9})  # font
    #
    #     # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown'])
    #
    #     plt.scatter(x=digits_tsne[i, 0],
    #                 y=digits_tsne[i, 1], # x, y
    #                 alpha=0.5,
    #                 color=colors[y_numpy[i]])

        # if is_known_numpy[i] == 0:
        #     plt.scatter(x=digits_tsne[i, 0],
        #                 y=digits_tsne[i, 1],  # x, y
        #                 alpha=0.5,
        #                 color=colors[10])
        # else:
        #     plt.scatter(x=digits_tsne[i, 0],
        #                 y=digits_tsne[i, 1],  # x, y
        #                 alpha=0.5,
        #                 color=colors[y_numpy[i]])

    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown'])
    # plt.xlim(digits_tsne[:, 0].min() - 10, digits_tsne[:, 0].max() + 10)  # 최소, 최대
    # plt.ylim(digits_tsne[:, 1].min() - 10, digits_tsne[:, 1].max() + 10)  # 최소, 최대
    # plt.xlabel('t-SNE x')  # x축 이름
    # plt.ylabel('t-SNE y')  # y축 이름
    # plt.show()  # 그래프 출력


if __name__ == '__main__':
    main()
