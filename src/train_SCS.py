import math
import os
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_reading_SCS import FashionDataset

import model_ResNet18 as Rn

import time
from torch.utils.tensorboard import SummaryWriter

# --------------------超参定义---------------------#
batch_size = 256  # 批量大小，每次输入的样本数量
learning_rate = 0  # 学习率,每次学习的更新幅度，学习率过大训练速度会更快，但结果可能不大稳定
use_lr_decay = 0  # 是否使用学习率衰减，是一种在训练过程中逐渐降低学习率的技术，帮助模型更好地收敛到最优解
lr_decay = 40  # 学习率衰减的周期，表示在多少个 epoch 之后进行一次学习率衰减
decay_rate = 0.5  # 学习率衰减的比率，表示每次衰减学习率时乘以的系数
num_epochs = 5  # 训练的总轮数，表示整个数据集将被用于训练多少次。lr_decay * 3 + 1

# 训练损失文件名
loss_file_name = 'train_loss_epoch' + str(num_epochs) + '_lr' + str(learning_rate) + '_lrdecay' + str(
    lr_decay) + 'b' + str(batch_size)
# 模型保存路径
model_save_path_ResNet18 = '../best_model_ResNet18_SCS'
model_save_path = ''

# 当前环境是否支持GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# 定义交叉熵损失函数，用于计算模型输出和真实标签之间的损失
error = nn.CrossEntropyLoss()

# ------------------读取创建训练数据集---------------------#
# 定义数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建数据集对象
train_set = FashionDataset(mode="train", total_samples=50000, transform=transform)
# 使用 DataLoader 进行批量加载
train_loader = DataLoader(
    train_set,
    batch_size=128,        # 每次训练加载 256 张
    shuffle=True,          # 每个 epoch 重新打乱数据,提高模型泛化能力
    num_workers=1,         # 使用 3个 CPU 线程加载数据
    pin_memory=False,      # 如果使用 GPU，建议设为 True
    prefetch_factor=2      # 预取数据，加快加载
)

test_set = FashionDataset(mode="test", total_samples=10000, transform=transform)
test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
    prefetch_factor=2
)

total_data_size = len(train_set)
total_batches = math.ceil(total_data_size / batch_size)


# -----------------------模型保存----------------------------#
# 在这里改保存路径
model_save_path = model_save_path_ResNet18
# 检查模型保存路径是否存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# -----------------------模型性能评估--------------------------#
# 计算模型在测试集或验证集上的性能，返回的是模型的准确率
def evaluate_model(model, device, data_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数
    total_loss = 0.0  # 记录总损失

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


# -----------------------训练过程--------------------------#
def exp_lr_scheduler(optimizer, epoch, decay_rate, init_lr, lr_decay):
    lr = init_lr * (decay_rate ** (epoch // lr_decay))

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train(new_learning_rate, my_model, use_lr_decay, lr_decay, decay_rate, num_epochs):
    graph_path = "../graph/SCS_resnet"
    # 获取当前时间
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_dir = os.path.join(graph_path, formatted_time)
    writer = SummaryWriter(log_dir=log_dir)
    # 记录模型结构
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy_input)

    global total_loss

    optimizer = torch.optim.Adam(my_model.parameters(), lr=new_learning_rate)  # 函数用于返回模型的所有可训练参数，lr决定了每次参数更新的步长
    my_model.train()
    best_accuracy = 0

    for epoch in range(1, num_epochs + 1):
        start_epoch_time = time.time()  # 记录每个epoch开始的时间

        loss_list = []
        precision_list = []
        total_hits = 0
        total_samples = 0
        total_loss = 0.0

        if use_lr_decay:
            optimizer = exp_lr_scheduler(optimizer, epoch, decay_rate, new_learning_rate, lr_decay)

        # 内层循环遍历训练数据加载器中的每个批次（batch）
        for batch_id, [images, labels] in enumerate(train_loader):
            # 前向传播
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            labels = labels.long()
            outputs = my_model(images)
            loss = error(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            predictions = torch.max(outputs, 1)[1]
            hit = (predictions == labels).sum().item()
            precision = hit / len(labels)

            # 计算每个epoch的总损失和准确率
            _, predicted = torch.max(outputs, 1)
            total_hits += hit
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        # 记录每个epoch的训练时间
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time

        writer.add_scalar("Time/Train/SCS", epoch_time, epoch)

        # 计算所有epoch的平均精度和损失并将结果保存到列表中
        epoch_accuracy = total_hits / total_samples
        epoch_loss = total_loss / total_samples

        writer.add_scalar("Loss/train/SCS", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train/SCS",  epoch_accuracy, epoch)

        # 用于打印每个训练周期的训练信息，包及括该epoch的loss和precision
        print(f'Epoch {epoch}: Avg Loss: {epoch_loss:.4f}, Avg Precision: {epoch_accuracy:.4f}')

        # 计算测试集 loss 和 accuracy
        test_accuracy = evaluate_model(my_model, device, test_loader, error)
        test_loss = evaluate_model(my_model, device, test_loader, error)

        writer.add_scalar("Loss/test/SCS", test_loss, epoch)

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

    writer.close()


if __name__ == '__main__':
    learning_rates = [0.001]
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
