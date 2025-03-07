import math
import os
# import time
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import methods as md
import data_reading as dr

import model_ResNet18 as Rn

# --------------------超参定义---------------------#
batch_size = 256  # 批量大小，每次输入的样本数量
learning_rate = 0  # 学习率,每次学习的更新幅度，学习率过大训练速度会更快，但结果可能不大稳定
use_lr_decay = 0  # 是否使用学习率衰减，是一种在训练过程中逐渐降低学习率的技术，帮助模型更好地收敛到最优解
lr_decay = 40  # 学习率衰减的周期，表示在多少个 epoch 之后进行一次学习率衰减
decay_rate = 0.5  # 学习率衰减的比率，表示每次衰减学习率时乘以的系数
num_epochs = 5  # 训练的总轮数，表示整个数据集将被用于训练多少次。lr_decay * 3 + 1

# root_path = '../cifar-10-batches-py/'  # 数据集路径
file_name1 = "data_batch_1"  # 修改不同的数据集
file_name2 = "test_batch"  # 测试集
# 训练损失文件名
loss_file_name = 'train_loss_epoch' + str(num_epochs) + '_lr' + str(learning_rate) + '_lrdecay' + str(
    lr_decay) + 'b' + str(batch_size)
# 模型保存路径
model_save_path_ResNet18 = '../best_model_ResNet18'
model_save_path = ''

# 当前环境是否支持GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# 字典，用于记录多个lr下，每个epoch中loss和precision在训练集上的表现
loss_data = {}
precision_data = {}

# 定义交叉熵损失函数，用于计算模型输出和真实标签之间的损失
error = nn.CrossEntropyLoss()

# ------------------读取创建训练数据集---------------------#
# 创建一个FashionDataset变量，表示类型是训练集，并转换为Pytorch的张量格式
# 使用DataLoader类将上一步创建的训练集train_set封装成一个批量生成器，每次从数据集中加载 batch_size 个样本
# # train_set = md.FashionDataset('train', root_path, file_name1, transform=transforms.ToTensor())
# train_set = dr.FashionDataset('train', file_name1, transform=transforms.ToTensor())
# train_loader = DataLoader(train_set, batch_size)

# # test_set = md.FashionDataset('test', root_path, file_name2, transform=transforms.ToTensor())
# test_set = dr.FashionDataset('test', file_name2, transform=transforms.ToTensor())
# test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# 创建一个FashionDataset变量，表示类型是训练集/测试集，并转换为Pytorch的张量格式
# 使用DataLoader类将上一步创建的训练集train_set封装成一个批量生成器，每次从数据集中加载 batch_size 个样本
# 定义数据增强
transform = transforms.Compose([
    transforms.ToTensor(),  # 数据预处理
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 创建数据集对象
train_set = dr.FashionDataset(mode="train", file_name=file_name1, transform=transform)
# 使用 DataLoader 进行批量加载
train_loader = DataLoader(
    train_set,
    batch_size=256,        # 每次训练加载 256 张
    shuffle=True,          # 每个 epoch 重新打乱数据,提高模型泛化能力
    num_workers=3,         # 使用 3个 CPU 线程加载数据
    pin_memory=False,      # 如果使用 GPU，建议设为 True
    prefetch_factor=2      # 预取数据，加快加载
)

# 创建数据集对象
test_set = dr.FashionDataset(mode="test", file_name=file_name2, transform=transform)
# 使用 DataLoader 进行批量加载
test_loader = DataLoader(
    test_set,
    batch_size=256,        # 每次训练加载 256 张
    shuffle=False,         # 每个 epoch 重新打乱数据,提高模型泛化能力
    num_workers=3,         # 使用 3个 CPU 线程加载数据
    pin_memory=False,      # 如果使用 GPU，建议设为 True
    prefetch_factor=2      # 预取数据，加快加载
)

# 假设 total_data_size 是你的数据集中的总样本数
total_data_size = len(train_set)
# 计算总批次数
total_batches = math.ceil(total_data_size / batch_size)
# 创建成功
# print('dataset created!')


# -----------------------模型保存----------------------------#
# 在这里改保存路径
model_save_path = model_save_path_ResNet18
# 检查模型保存路径是否存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)  # 如果模型保存路径不存在，则创建


# -----------------------模型性能评估--------------------------#
# 计算模型在测试集或验证集上的性能，返回的是模型的准确率
def evaluate_model(model, device, data_loader):
    model.eval()  # 将模型设置为评估模式
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数
    with torch.no_grad():    # 关闭反向传播，因为在评估过程中，不需要进行反向传播或更新权重
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# -----------------------训练过程--------------------------#
# 学习率调度器函数，参数分别是优化器对象、当前训练的轮数、初始学习率、学习率衰减周期
def exp_lr_scheduler(optimizer, epoch, decay_rate, init_lr, lr_decay):
    # 计算当前伦次下的学习率
    lr = init_lr * (decay_rate ** (epoch // lr_decay))
    # 如果当前轮次是学习率衰减周期的整数倍，即到达了一个学习率更新点
    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))
    # 遍历优化器中的参数数组，然后更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回优化器对象
    return optimizer


def train(new_learning_rate, my_model, use_lr_decay, lr_decay, decay_rate, num_epochs):
    global total_loss
    # 定义一个Adam优化器，用于更新模型参数以最小化损失函数
    optimizer = torch.optim.Adam(my_model.parameters(), lr=new_learning_rate)  # 函数用于返回模型的所有可训练参数，lr决定了每次参数更新的步长
    my_model.train()  # 将模型设置为训练模式
    best_accuracy = 0  # 记录模型在训练过程中测试集上的最佳准确率，初始值为0
    # 在每个epoch结束后，保存该轮次的平均loss和平均precision值
    avg_loss_per_epoch = []
    avg_precision_per_epoch = []
    for epoch in range(1, num_epochs + 1):
        loss_list = []  # 记录当前 epoch 中每个batch的loss值
        precision_list = []  # 记录当前 epoch 中每个batch的accuracy值
        total_hits = 0  # 统计当前 epoch 中所有批次的正确预测（hit）总数
        total_samples = 0  # 统计当前 epoch 中所有批次的总样本数
        total_loss = 0.0  # 累计当前 epoch 中所有批次的总损失值
        # 若开启学习率衰减，则根据当前epoch来调整学习率
        if use_lr_decay:
            optimizer = exp_lr_scheduler(optimizer, epoch, decay_rate, new_learning_rate, lr_decay)
        # 记录每个训练周期的开始时间
        # start_time = time.time()

        # 内层循环遍历训练数据加载器中的每个批次（batch）
        for batch_id, [images, labels] in enumerate(train_loader):
            # 记录每个批次的开始时间
            # batch_time = time.time()
            # print('batch {} processed!'.format(batch_id))
            # import ipdb; ipdb.set_trace()
            # 前向传播
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            labels = labels.long()  # 调整标签类型
            outputs = my_model(images)  # 得到模型的预测输出
            loss = error(outputs, labels)  # 计算损失值
            loss_list.append(loss.item())  # 将损失值添加到损失列表中
            optimizer.zero_grad()  # 优化器梯度清零

            # 反向传播
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # torch.max(outputs, 1) 返回第一个元素是每个样本预测输出的最大值
            # 第二个元素是每个样本预测输出的最大值所在的索引。通过 [1] 取出最大值所在的索引，即模型预测的类别。
            predictions = torch.max(outputs, 1)[1]
            # 统计预测正确的数量，使用.item() 方法将统计结果转换为Python中的标量值
            hit = (predictions == labels).sum().item()
            # 预测准确率
            precision = hit / len(labels)
            # 将当前批次的精度值添加到precision_list中，以便后续分析和可视化
            precision_list.append(precision)

            # 计算每个epoch的总损失和准确率
            _, predicted = torch.max(outputs, 1)  # 由于只需要索引（即预测类别），所以使用_来忽略 torch.max 返回的第一个元素（即最大值）
            total_hits += hit
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        # 计算所有epoch的平均精度和损失并将结果保存到列表中
        epoch_accuracy = total_hits / total_samples
        epoch_loss = total_loss / total_samples
        avg_loss_per_epoch.append(epoch_loss)
        avg_precision_per_epoch.append(epoch_accuracy)

        # 用于打印每个训练周期的训练信息，包及括该epoch的loss和precision
        print(f'Epoch {epoch}: Avg Loss: {epoch_loss:.4f}, Avg Precision: {epoch_accuracy:.4f}')

        test_accuracy = evaluate_model(my_model, device, test_loader)
        model_path = os.path.join(model_save_path,
                                  f'lr_{new_learning_rate}_use_lr_decay{use_lr_decay}_lrdecay_{lr_decay}_decay_rate{decay_rate}_bsize{batch_size}_num_epochs{num_epochs}.pth')

        print(f"Test accuracy after epoch {epoch}: {test_accuracy}")
        # 保存在测试集上表现最好的模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = model_path
            for file in os.listdir(model_save_path):
                if file.endswith(".pth"):
                    os.remove(os.path.join(model_save_path, file))  # 如果路径存在，删除该文件夹中的所有.pth文件
            torch.save(my_model.state_dict(), best_model_path)  # 保存模型结构和参数

    # 保存每个学习率下的损失和精度数据，便于后续可视化
    loss_data[new_learning_rate] = avg_loss_per_epoch
    precision_data[new_learning_rate] = avg_precision_per_epoch
    # md.draw(avg_loss_per_epoch, avg_precision_per_epoch, num_epochs, learning_rate, lr_decay)


# -----------------------实现多组超参同时训练--------------------------#
# # 不开学习率
# if __name__ == '__main__':
    # learning_rates = [0.001]
    # num_epochs = [15]
    # param_combinations2 = itertools.product(learning_rates, num_epochs)
    # for lr, num_epoch in param_combinations2:
    #     print(
    #         f"Training with lr={lr}, use_lr_decay={0}, lr_decay={0}, decay_rate={0}, num_epochs={num_epoch},
    #         batch_Size={batch_size}")

    #     # model =Rn.ResNet18()

    #     if cuda:
    #         model = model.cuda()
    #     train(lr, model, 0, 0, 0, num_epoch)

# 开学习率
if __name__ == '__main__':
    learning_rates = [0.001,0.0005]
    use_lr_decay = [1]
    lr_decays = [15]
    decay_rates = [0.5]
    num_epochs = [50]
    param_combinations1 = itertools.product(learning_rates, lr_decays, decay_rates, num_epochs)
    if use_lr_decay != 0:
        for lr, lr_decay, decay_rate, num_epoch in param_combinations1:
            print(
                f"Training with lr={lr}, use_lr_decay={use_lr_decay}, lr_decay={lr_decay}, decay_rate={decay_rate}, "
                f"batch_Size={batch_size}, num_epochs={num_epoch}")
            model =Rn.ResNet18()
            if cuda:
                model = model.cuda()
            train(lr, model, use_lr_decay, lr_decay, decay_rate, num_epoch)
