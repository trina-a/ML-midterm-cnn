from cnn_function import *
import numpy as np
import torch
import torchvision  # 数据集、数据预处理等
import torchvision.transforms as transforms  # 数据预处理
import torch.nn as nn  # 网络结构
import torch.optim as optim  # 优化器 比如SGD
import json
import torch.utils.data as data_utils  # 对数据集进行分批加载的工具集
import torchvision.utils as utils
import copy


######################### 设置参数和下载数据
batch_size = 64
C, W, H = 3, 32, 32
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 如果有gpu使用gpu，否则cpu
new_transform = transforms.Compose([  # 数据预处理
    transforms.ToTensor(),  # 转化成张量（类似于numpy的array,是pytorch中的基础数据结构），并且归一化（0，255）=>（0，1）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 数据归一化到（-1，1）范围，
])

train_set = torchvision.datasets.CIFAR100('data',  # 相对路径
                                          train=True,  # True：获取训练集  False：获取测试集
                                          download=True,  # 是否需要下载
                                          transform=new_transform  # 使用之前定义好的数据预处理
                                          )
test_set = torchvision.datasets.CIFAR100('data', train=False, download=True, transform=new_transform)

train_loader = torch.utils.data.DataLoader(train_set,  # 批处理数据,转化成迭代器,共784个，每个大小64
                                           batch_size=batch_size,  # mini-batch的大小
                                           shuffle=True,  # 是否打乱顺序
                                           num_workers=0,  # 多核运作，可以加快处理速度，看电脑的处理器核心数
                                           )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
print(train_set)
print("size=", len(train_set))
print("")
print(test_set)
print("size=", len(test_set))

################################### 画出9张图像
def imageSavePIL(images,fileName,normalization=True,mean=0,std=1):
    image=utils.make_grid(images)
    #是否原图进行了normalization
    if normalization:
        #交换之后，(H,W,C)
        image=image.permute(1,2,0)
        image=(image*torch.tensor(std)+torch.tensor(mean))
        #交换之后,(C,H,W)
        image=image.permute(2,0,1)
    #将tensor转化为Image格式
    image=transforms.ToPILImage()(image)
    #存储图片
    image.save(fileName)

for _, (images, labels) in enumerate(train_loader):
    if _ > 2:
        break

    # mixup
    alpha = 1
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size()[0])
    mixed_image = lam * images + (1 - lam) * images[index, :]
    imageSavePIL(mixed_image[0], str(_) + "mixup.png", std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])

    # cutmix
    beta = 1
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    cutmix_image = copy.deepcopy(images)
    cutmix_image[:, :, bbx1:bbx2, bby1:bby2] = cutmix_image[index, :, bbx1:bbx2, bby1:bby2]
    imageSavePIL(cutmix_image[0], str(_) + "cutmix.png", std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
    # 如果是mixout：inputs[:, :, bbx1:bbx2, bby1:bby2] =0

    # cutout
    cutout_image = copy.deepcopy(images)
    cutout_image[:, :, bbx1:bbx2, bby1:bby2] = 255
    imageSavePIL(cutout_image[0], str(_) + "cutout.png", std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])

#####################################训练模型
def train(method='baseline', epochs=50):
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    print("start train!")
    for epoch in range(epochs):
        running_loss = 0.0
        # 迭代，批次训练
        for i, data in enumerate(train_loader):
            print(F"epoch: {epoch}, batch: {i}")
            # 获取训练数据
            inputs, labels = data[0], data[1]
            # 注意，pytorch张量分gpu和cpu，如果网络是gpu，数据是cpu会报错，所以要加下面这行
            inputs, labels = inputs.to(device), labels.to(device)
            # 每个batch都要清零，否则反向传播的梯度会一直累加
            optimizer.zero_grad()
            # loss function
            criterion = nn.CrossEntropyLoss()

            if method == 'baseline':
                # 正向传播
                outputs = cnn_net.forward(inputs)
                # 计算损失值
                loss = loss_function(outputs, labels)
            elif method == 'mixup':
                mixup_alpha = 1
                mixed_image, label_a, label_b, lam = mixup_data(inputs, labels, mixup_alpha)
                outputs = cnn_net.forward(mixed_image)
                loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
            elif method == 'cutmix' or method == 'cutout':
                beta = 1
                cutmix_prob = 0.5
                r = np.random.rand(1)  # 不是每个epoch都进行cutmix
                if beta > 0 and r < cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(inputs.size()[0])

                    target_a, target_b = labels, labels[rand_index]

                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    if method == 'cutmix':
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    else:
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = 0

                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    # compute output
                    outputs = cnn_net.forward(inputs)
                    # output = model(inputs)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                else:
                    # compute output
                    outputs = cnn_net.forward(inputs)
                    # output = model(inputs)
                    loss = criterion(outputs, labels)
            predict_labels = outputs.argmax(1)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 损失值累加
            running_loss += loss.item()
            # 记录损失值
            loss_history.append(loss.item())
            # 记录accuracy
            new = (predict_labels == labels)
            minibatch_acc = np.array(new, dtype="float64").mean()
            train_acc_history.append(minibatch_acc)

            # 每100个mini-batch显示一次损失值
            if i % 100 == 99:
                print('epoch: %d  minibatch: %d ====== mean loss:%.3f  train_accuracy:%.4f' % (
                epoch + 1, i + 1, running_loss / 100, minibatch_acc))
                running_loss = 0.0

        # 每个epoch输出一个test accuracy
        acc = 0.0
        for k in test_loader:
            test_inputs = k[0]
            predict = cnn_net.forward(test_inputs.to(device))
            predict = predict.argmax(1)
            lab = k[1]
            new = (predict == lab)
            acc += np.array(new, dtype="float64").mean()
        test_acc_history.append(acc / len(test_loader))
        print('test_accuracy:%.4f' % (acc / len(test_loader)))

    print('Finished Training')

    return loss_history, train_acc_history, test_acc_history

# cutmix
# 部署模型
cnn_net = AlexNet()
cnn_net.to(device)
loss_function = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.Adam(cnn_net.parameters(), lr=0.0008, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 训练
loss_history, train_acc_history, test_acc_history = train(method='cutmix', epochs=50)
# 保存数据
filename1 = 'cutmix_loss_history.json'
with open(filename1, 'w') as file_obj:
    json.dump(loss_history, file_obj)
filename2 = 'cutmix_train_acc_history.json'
with open(filename2, 'w') as file_obj:
    json.dump(train_acc_history, file_obj)
filename3 = 'cutmix_test_acc_history.json'
with open(filename3, 'w') as file_obj:
    json.dump(test_acc_history, file_obj)
torch.save(cnn_net, 'cutmix.pth')  # 保存模型

# cutout
# 部署模型
cnn_net = AlexNet()
cnn_net.to(device)
loss_function = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.Adam(cnn_net.parameters(), lr=0.0008, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 训练
loss_history, train_acc_history, test_acc_history = train(method='cutout', epochs=50)
# 保存数据
filename1 = 'cutout_loss_history.json'
with open(filename1, 'w') as file_obj:
    json.dump(loss_history, file_obj)
filename2 = 'cutout_train_acc_history.json'
with open(filename2, 'w') as file_obj:
    json.dump(train_acc_history, file_obj)
filename3 = 'cutouttest_acc_history.json'
with open(filename3, 'w') as file_obj:
    json.dump(test_acc_history, file_obj)
torch.save(cnn_net, 'cutout.pth')  # 保存模型

# mixup
# 部署模型
cnn_net = AlexNet()
cnn_net.to(device)
loss_function = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.Adam(cnn_net.parameters(), lr=0.0008, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 训练
loss_history, train_acc_history, test_acc_history = train(method='mixup', epochs=50)
# 保存数据
filename1 = 'mixup_loss_history.json'
with open(filename1, 'w') as file_obj:
    json.dump(loss_history, file_obj)
filename2 = 'mixup_train_acc_history.json'
with open(filename2, 'w') as file_obj:
    json.dump(train_acc_history, file_obj)
filename3 = 'mixup_test_acc_history.json'
with open(filename3, 'w') as file_obj:
    json.dump(test_acc_history, file_obj)
torch.save(cnn_net, 'mixup.pth')  # 保存模型

# baseline
# 部署模型
cnn_net = AlexNet()
cnn_net.to(device)
loss_function = nn.CrossEntropyLoss() #交叉熵
optimizer = optim.Adam(cnn_net.parameters(),lr=0.0005,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
# 训练
loss_history,train_acc_history,test_acc_history=train(method='baseline',epoch=50)
# 保存数据
filename1='base_loss_history.json'
with open(filename1,'w') as file_obj:
    json.dump(loss_history,file_obj)
filename2='base_train_acc_history.json'
with open(filename2,'w') as file_obj:
    json.dump(train_acc_history,file_obj)
filename3='base_test_acc_history.json'
with open(filename3,'w') as file_obj:
    json.dump(test_acc_history,file_obj)
torch.save(cnn_net, 'baseline.pth') #保存模型