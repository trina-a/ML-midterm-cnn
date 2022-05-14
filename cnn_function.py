import numpy as np
import torch
import torch.nn as nn  # 网络结构


# cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# mixup
def mixup_data(image, label, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]
    label_a, label_b = label, label[index]
    return mixed_image, label_a, label_b, lam


def mixup_criterion(criterion, prediction, label_a, label_b, lam):
    """
    :param criterion: the cross entropy criterion
    :param prediction: y_pred
    :param label_a: label = lam * label_a + (1-lam)* label_b
    :param label_b: label = lam * label_a + (1-lam)* label_b
    :param lam: label = lam * label_a + (1-lam)* label_b
    :return:  cross_entropy(pred,label)
    """
    return lam * criterion(prediction, label_a) + (1 - lam) * criterion(prediction, label_b)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2),
            # 论文中kernel_size = 11,stride = 4,padding = 2
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # a为归一化之前的神经元，b为归一化之后的神经元；N是卷积核的个数，也就是生成的FeatureMap的个数；k，α，β，n是超参数，论文中使用的值是k=2，n=5，α=0.0001，β=0.75。
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # a为归一化之前的神经元，b为归一化之后的神经元；N是卷积核的个数，也就是生成的FeatureMap的个数；k，α，β，n是超参数，论文中使用的值是k=2，n=5，α=0.0001，β=0.75。
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 256*4*4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # OUT : 384*2*2
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=100),
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.view(-1, 384 * 2 * 2)
        x = self.linear(x)
        return x
