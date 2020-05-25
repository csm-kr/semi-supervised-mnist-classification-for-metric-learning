from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as tfs


class SAMPLE_MNIST(Dataset):
    def __init__(self, semi_mnist, transform):
        super().__init__()
        self.semi_mnist = semi_mnist
        self.known_idx = self.semi_mnist.known_idx
        self.data = self.semi_mnist.data[self.known_idx]        # uint8 : ByteTensor
        self.targets = self.semi_mnist.targets[self.known_idx]  # int64 : Longtensor
        self.transform = transform

    def __getitem__(self, index):
        """
        random 으로 100 개만 label 이 존재하는 dataset 만들어서 고정시켜 뽑기..!
        """
        images = self.data[index]
        targets = self.targets[index]

        images = images.float().div(255).unsqueeze(0)

        if self.transform is not None:
            images = self.transform(images)

        return images, targets

    def __len__(self):

        return len(self.data)


if __name__ == '__main__':

    from dataset import SEMI_MNIST
    mean, std = 0.1307, 0.3081

    transform = tfs.Compose([tfs.Normalize((mean,), (std,))])

    # len of train mnist dataset is 6000
    train_set = MNIST('./data/MNIST',
                      train=True,
                      download=True)

    train_set = SEMI_MNIST(train_set, transform=transform)
    train_set = SAMPLE_MNIST(train_set, transform)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set,
                              batch_size=2,
                              shuffle=True
                              )

    for (img, targets) in train_loader:
        print(img.type(), img.size())
        print(targets.type(), targets.size())

        import cv2
        import matplotlib.pyplot as plt
        # 1. img val
        img_numpy = img[0].squeeze().numpy()
        plt.imshow(img_numpy, cmap='Greys')
        plt.show()
        # cv2.imshow('output', img_numpy)
        # 2. target val
        print(targets[0].item())
        cv2.waitKey(0)




