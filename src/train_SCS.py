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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch.optim.lr_scheduler as lr_scheduler

from tensorboard import program

# --------------------超参定义---------------------#

# 模型保存路径
model_save_path_ResNet18 = '../best_model_ResNet18_SCS'
model_save_path = ''

# 当前环境是否支持GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

is_training = 1

# 定义交叉熵损失函数，用于计算模型输出和真实标签之间的损失
error = nn.CrossEntropyLoss()

# ------------------读取创建训练数据集---------------------#
# 创建一个全局 requests.Session 并设置连接池大小
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100,
                                        max_retries=Retry(total=3, backoff_factor=0.1))  # 连接池大小 100
session.mount("http://", adapter)
# 定义数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


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
def launch_tensorboard(log_dir):
    """在PyCharm中通过Python代码启动TensorBoard"""
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir, "--port", "6006"])
    url = tb.launch()
    print(f"TensorBoard已启动，访问: {url}")


def train(batch_size, new_learning_rate, my_model, lr_decay, decay_rate, num_epochs, training_state=None):
    # 创建数据集对象
    train_set = FashionDataset(mode="train", total_samples=50000, transform=transform, session=session)
    # 使用 DataLoader 进行批量加载
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,  # 每次训练加载 256 张
        shuffle=True,  # 每个 epoch 重新打乱数据,提高模型泛化能力
        num_workers=1,  # 使用 3个 CPU 线程加载数据
        pin_memory=False,  # 如果使用 GPU，建议设为 True
        prefetch_factor=2  # 预取数据，加快加载
    )

    test_set = FashionDataset(mode="test", total_samples=10000, transform=transform, session=session)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=2
    )

    graph_path = "../graph/SCS_resnet"
    # 获取当前时间
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_dir = os.path.join(graph_path, formatted_time)

    # graph_path = "../tensorboard"
    # if os.path.exists(graph_path):
    #     shutil.rmtree(graph_path)
    #     os.makedirs(graph_path, exist_ok=True)
    # log_dir = os.path.join(graph_path)

    writer = SummaryWriter(log_dir=log_dir)
    # 记录模型结构
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(my_model, dummy_input)
    launch_tensorboard(log_dir)

    global total_loss

    optimizer = torch.optim.Adam(my_model.parameters(), lr=new_learning_rate)  # 函数用于返回模型的所有可训练参数，lr决定了每次参数更新的步长
    # 添加学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=decay_rate)
    my_model.train()
    best_accuracy = 0

    patience = 5
    no_improve_count = 0

    for epoch in range(1, num_epochs + 1):
        if not training_state['is_training']:
            break

        start_epoch_time = time.time()  # 记录每个epoch开始的时间

        total_hits = 0
        total_samples = 0
        total_loss = 0.0

        # 内层循环遍历训练数据加载器中的每个批次（batch）
        for batch_id, [images, labels] in enumerate(train_loader):
            if not training_state['is_training']:
                break

            images, labels = images.to(device), labels.to(device)
            labels = labels.long()

            outputs = my_model(images)
            loss = error(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            predictions = torch.max(outputs, 1)[1]
            hit = (predictions == labels).sum().item()

            # 计算每个epoch的总损失和准确率
            _, predicted = torch.max(outputs, 1)
            total_hits += hit
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        # 更新学习率
        scheduler.step()  # 在每个 epoch 结束时调整学习率

        training_state["epoch"] = epoch

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
        test_accuracy, test_loss = evaluate_model(my_model, device, test_loader, error)

        writer.add_scalar("Accuracy/test/distribute", test_accuracy, epoch)
        writer.add_scalar("Loss/test/SCS", test_loss, epoch)

        model_path = os.path.join(model_save_path,
                                  f'lr_{new_learning_rate}_lrdecay_{lr_decay}_decay_rate{decay_rate}_bsize{batch_size}_num_epochs{num_epochs}.pth')

        print(f"Test accuracy after epoch {epoch}: {test_accuracy}")

        # 保存在测试集上表现最好的模型
        if test_accuracy > best_accuracy:
            no_improve_count = 0
            best_accuracy = test_accuracy
            best_model_path = model_path
            for file in os.listdir(model_save_path):
                if file.endswith(".pth"):
                    os.remove(os.path.join(model_save_path, file))  # 如果路径存在，删除该文件夹中的所有.pth文件
            torch.save(my_model.state_dict(), best_model_path)  # 保存模型结构和参数
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    writer.close()


def train_resnet(num_epoch=10, training_state=None):
    batch_sizes = [128]
    learning_rates = [0.0005]
    lr_decays = [15]
    decay_rates = [0.5]
    training_state["epoch"] = 0
    training_state["max_epoch"] = num_epoch  # 保持一致
    param_combinations1 = itertools.product(batch_sizes, learning_rates, lr_decays, decay_rates)
    for batch_size, lr, lr_decay, decay_rate in param_combinations1:
        print(
            f"Training with lr={lr}, lr_decay={lr_decay}, decay_rate={decay_rate}, "
            f"batch_Size={batch_size}, num_epochs={num_epoch }")
        model =Rn.ResNet18()
        if cuda:
            model = model.cuda()
        train(batch_size, lr, model, lr_decay, decay_rate, num_epoch, training_state=training_state)
        training_state["is_training"] = False


if __name__ == '__main__':
    training_state = {
        "is_training": True,
        "epoch": 0,
        "max_epoch": 30
    }
    train_resnet(30, training_state)
    # learning_rates = [0.0005]
    # use_lr_decay = [1]
    # lr_decays = [15]
    # decay_rates = [0.5]
    # num_epochs = [50]
    # param_combinations1 = itertools.product(learning_rates, lr_decays, decay_rates, num_epochs)
    # if use_lr_decay != 0:
    #     for lr, lr_decay, decay_rate, num_epoch in param_combinations1:
    #         print(
    #             f"Training with lr={lr}, use_lr_decay={use_lr_decay}, lr_decay={lr_decay}, decay_rate={decay_rate}, "
    #             f"batch_Size={batch_size}, num_epochs={num_epoch}")
    #         model =Rn.ResNet18()
    #         if cuda:
    #             model = model.cuda()
    #         train(lr, model, use_lr_decay, lr_decay, decay_rate, num_epoch)
