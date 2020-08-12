import torch.nn as nn
import torch.nn.functional as F
import torch


class MetricCrossEntropy(nn.Module):
    def __init__(self):
        """
        cross entropy for metric learning

        @param img: torch.ByteTensor : [N, 2] : tensor of forwarding images
        @param targets: torch.LongTensor : [N] : tensor of targets
        @param samples: torch.ByteTensor : [N, 10, 2] : tensor of forwarding samples
        @param is_known: torch.LongTensor : [N] : tensor of letting me teach "is_known"
        @return:
        """
        super(MetricCrossEntropy, self).__init__()

    # ORIGINAL
    # def forward(self, x, y, z, k, lambda_1=1, lambda_2=1):
    #     # F.mse_loss(x, z)
    #     batch_size = x.size(0)
    #     m_softmax = self.metric_soft_max(x, z)  # [N, 10]
    #     known = k.unsqueeze(-1)
    #     unknown = (1-k).unsqueeze(-1)
    #     one_hot = torch.zeros([batch_size, 10]).cuda()
    #     one_hot = one_hot.scatter(1, y.view(-1, 1), 1)  # convert target to one-hot [N, 10]
    #     pred_label = m_softmax * unknown
    #     m = -1 * torch.log(m_softmax)  # [N, 10]
    #     # y # [N]
    #     # m 에서 m[i][y[i]] 를 하고싶습니다.  --> gather, scatter_, dim=1 인것이 중요!!
    #     label_loss = known * torch.gather(m, 1, y.unsqueeze(-1))  # [32, 1]
    #     unlabel_loss = -1 * (pred_label * torch.log(m_softmax)).sum(dim=1)
    #     loss = lambda_1 * label_loss + lambda_2 * unlabel_loss
    #     loss = loss.mean()
    #
    #     return loss

    # CUSTOM
    def forward(self, x, y, z=None, k=None, lambda_1=1, lambda_2=1):
        batch_size = x.size(0)

        m_softmax = self.metric_soft_max(x, z)  # [N, 10]
        m_softmax = torch.pow(m_softmax, 3)
        m_softmax_ = torch.pow(1 - m_softmax, 2)
        known = k.unsqueeze(-1)
        unknown = (1-k).unsqueeze(-1)

        one_hot = torch.zeros([batch_size, 10]).cuda()
        one_hot = one_hot.scatter(1, y.view(-1, 1), 1)  # convert target to one-hot [N, 10]

        label = one_hot * known
        pred_label = m_softmax * unknown

        unlabel_loss = -1 * (m_softmax_ * pred_label * torch.log(m_softmax)).sum(dim=1)
        label_loss = -1 * (label * torch.log(F.softmax(x, dim=1))).sum(dim=1)

        loss = lambda_1 * label_loss + lambda_2 * unlabel_loss
        loss = loss.mean()
        return loss

    # custom 2

    # def forward(self, x, y, z=None, k=None, lambda_1=1, lambda_2=1):
    #     batch_size = x.size(0)
    #     m_softmax = self.metric_soft_max(x, z)  # [N, 10]
    #     known = k.unsqueeze(-1)
    #     unknown = (1-k).unsqueeze(-1)
    #     one_hot = torch.zeros([batch_size, 10]).cuda()
    #     one_hot = one_hot.scatter(1, y.view(-1, 1), 1)  # convert target to one-hot [N, 10]
    #     label = one_hot * known
    #     # print(torch.equal(label, one_hot))  # 같음
    #     pred_label = m_softmax * unknown
    #     unlabel_loss = -1 * (pred_label * torch.log(m_softmax)).sum(dim=1)
    #     label_loss = -1 * (label * torch.log(F.softmax(x, dim=1))).sum(dim=1)
    #     # 2
    #     loss = -1 * ((lambda_1 * label + lambda_2 * pred_label) * torch.log(m_softmax)).sum(dim=1)
    #     loss = loss.mean()
    #     return loss

    def metric_soft_max(self, x, z):
        """
        @ x : [N, 2]
        @ z : [N, 10, 2]
        @param : e - || F(x) - F(z_i) || ^ 2
        """
        x = x.unsqueeze(1)       # N, 1, 2
        exp = torch.exp(-1 * (x - z).pow(2).sum(dim=-1).sqrt())  # N, 10
        exp_sum = torch.sum(exp, dim=1, keepdim=True)
        return exp / exp_sum


if __name__ == '__main__':

    batch_size = 5
    imgs = torch.randn([batch_size, 2])
    targets = torch.randint(10, [batch_size])
    samples = torch.randn([batch_size, 10, 2])
    is_known = torch.randint(2, [batch_size])

    criterion = MetricCrossEntropy()
    loss = criterion(imgs, targets, samples, is_known)
    print(loss)
