import torch.nn as nn
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 16, 5, padding=1),  # 28
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(16),
                                     nn.MaxPool2d(2, stride=2),  # 13

                                     nn.Conv2d(16, 32, 3, padding=1),  # 6
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32),

                                     nn.Conv2d(32, 64, 3, padding=1),  # 6
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),

                                     nn.Conv2d(64, 64, 3, padding=1),  # 6
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2, stride=2),  # 3

                                     nn.Conv2d(64, 128, 3, padding=1),  # 3
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(128),
                                     # nn.AvgPool2d((6, 6))
                                     )

        self.fc = nn.Sequential(nn.Linear(128 * 6 * 6, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 10),
                                )

    def forward(self, x):
        output = self.convnet(x)
        # print(output.size())
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    img = torch.randn([10, 1, 28, 28]).cuda()
    network = EmbeddingNet().cuda()
    print(network(img).size())


