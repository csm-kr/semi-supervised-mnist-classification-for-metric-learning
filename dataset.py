from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import torch
import numpy as np
from PIL import Image


class SEMI_MNIST(Dataset):
    def __init__(self, mnist, transform, num_samples=600):
        super().__init__()
        self.mnist = mnist
        self.data = self.mnist.data        # uint8 : ByteTensor
        self.targets = self.mnist.targets  # int64 : Longtensor
        self.num_samples = num_samples
        self.transform = transform

        known_idx = []
        unknown_idx = []
        self.known_label = torch.empty([10, self.num_samples])  # 10 * num_samples 개

        for i in range(10):
            k_idx = torch.where(self.targets == i)[0][:self.num_samples]
            known_idx.append(k_idx)
            unknown_idx.append(torch.where(self.targets == i)[0][self.num_samples:])
            self.known_label[i] = k_idx

        self.known_idx = torch.cat(known_idx, dim=0).numpy()
        self.unknown_idx = torch.cat(unknown_idx, dim=0).numpy()  # tensor to numpy
        self.known_label = self.known_label.numpy()  # 10, 500

    def __getitem__(self, index):
        """
        random 으로 100 개만 label 이 존재하는 dataset 만들어서 고정시켜 뽑기..!
        """

        images = self.data[index]
        targets = self.targets[index]

        images = images.float().div(255).unsqueeze(0)

        sample_idx_r = np.random.randint(self.num_samples, size=10)  #
        samples_idx = np.ndarray([10])

        # get the samples for uniform distributions
        for i in range(10):
            samples_idx[i] = self.known_label[i][sample_idx_r[i]]

        samples = self.data[samples_idx]
        samples = samples.float().div(255)

        is_known = 0
        if index in self.known_idx:
            is_known = 1

        if self.transform is not None:
            images = self.transform(images)
            samples = self.transform(samples)
        # torch.Size([B, 1, 28, 28])
        # torch.Size([B)
        # torch.Size([B, 10, 28, 28])
        # torch.Size([B])

        return images, targets, samples, is_known

    def __len__(self):
        # return self.num_samples * 10
        return len(self.data)


if __name__ == '__main__':

    mean, std = 0.1307, 0.3081

    transform = tfs.Compose([tfs.ToTensor()])

    # len of train mnist dataset is 60000
    train_set = MNIST('./data/MNIST',
                      train=True,
                      download=True)

    train_set = SEMI_MNIST(train_set, transform=transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set,
                              batch_size=2,
                              shuffle=True
                              )

    for (img, targets, samples, is_known) in train_loader:
        print(img.type(), img.size())
        print(targets.type(), targets.size())
        print(samples.type(), samples.size())
        print(is_known.type(), is_known.size())

        import cv2
        import matplotlib.pyplot as plt
        # 1. img val
        img_numpy = img[0].squeeze().numpy()
        plt.imshow(img_numpy, cmap='Greys')
        plt.show()
        # cv2.imshow('output', img_numpy)

        # 2. target val
        print(targets[0].item())

        # 3. sample val
        samples = samples.numpy()
        fig1 = plt.figure(1)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(samples[0][i], cmap='Greys')
        plt.show()
        cv2.waitKey(0)




