import os
import itertools
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from data_reading_SCS import FashionDataset
import model_ResNet18 as Rn

import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------超参数搜索空间---------------------#
batch_sizes = [128]
learning_rates = [0.0001]
lr_decays = [15]
decay_rates = [0.5]
num_epochs_list = [50]  # 不同超参数组合的训练轮数

# 生成所有超参数组合
param_combinations = list(itertools.product(batch_sizes, learning_rates, lr_decays, decay_rates, num_epochs_list))


model_save_path_ResNet18 = '../best_model_ResNet18_distribute'
model_save_path = model_save_path_ResNet18

# 检查模型保存路径是否存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path,)  # 如果模型保存路径不存在，则创建


# -----------------分布式训练设置-----------------#
def setup():
    """初始化分布式环境"""
    # master_addr = os.environ.get("MASTER_ADDR", "10.27.251.68")  # 获取主节点 IP
    # master_port = os.environ.get("MASTER_PORT", "56834")  # 获取主节点端口
    #
    # os.environ["MASTER_ADDR"] = master_addr
    # os.environ["MASTER_PORT"] = master_port

    master_addr = os.environ["MASTER_ADDR"]  # 直接读取命令行传递的环境变量
    master_port = os.environ["MASTER_PORT"]
    import socket
    print(f"Resolved IP: {socket.gethostbyname(socket.gethostname())}")

    rank = int(os.environ["RANK"])  # 获取进程的 rank
    world_size = int(os.environ["WORLD_SIZE"])  # 获取全局进程数

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"[INFO] Process {rank}/{world_size} initialized on {master_addr}:{master_port}")
    return rank, world_size


def cleanup():
    """清理分布式进程"""
    dist.destroy_process_group()


# ------------------训练函数------------------#
def train():
    rank, world_size = setup()  # 初始化分布式环境

    # 设备设置
    device = torch.device("cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 遍历所有超参数组合
    for param_id, (batch_size, learning_rate, lr_decay, decay_rate, num_epochs) in enumerate(param_combinations):
        print(f"[Rank {rank}] Training with batch_size={batch_size}, lr={learning_rate}, "
              f"lr_decay={lr_decay}, decay_rate={decay_rate}, epochs={num_epochs}")

        # 创建一个全局 requests.Session 并设置连接池大小
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100,
                                                max_retries=Retry(total=3, backoff_factor=0.1))  # 连接池大小 100
        session.mount("http://", adapter)

        # 训练数据集
        # train_set = FashionDataset(mode="train", file_name=file_name1, transform=transform)
        train_set = FashionDataset(mode="train", total_samples=50000, transform=transform, session=session)
        # 为了能够按顺序划分数据子集，拿到不同部分数据，所以数据集不能够进行随机打散，所以DataLoader用参数 'shuffle': False
        # 而在为了在每个 epoch 随机重新分配数据，避免不同 rank 之间的数据模式固定，提高泛化能力 DistributedSampler 用参数 'shuffle': True
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, prefetch_factor=2,
                                  pin_memory=False, sampler=train_sampler)

        # 测试数据集
        # test_set = FashionDataset(mode="test", file_name=file_name2, transform=transform)
        test_set = FashionDataset(mode="test", total_samples=10000, transform=transform, session=session)
        # 设定 shuffle=False 保证 每次评估时数据顺序相同，确保实验的可复现性
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1, prefetch_factor=2,
                                 pin_memory=False, sampler=test_sampler)

        # 初始化模型
        model = Rn.ResNet18().to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 添加学习率调度器
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=decay_rate)

        # 仅在 rank == 0 时初始化 TensorBoard
        if rank == 0:
            graph_path = "../graph/distributed_resnet"
            # 获取当前时间
            current_time = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
            log_dir = os.path.join(graph_path, formatted_time)
            writer = SummaryWriter(log_dir=log_dir)
            # 记录模型结构
            dummy_input = torch.randn(1, 3, 32, 32).to(device)
            writer.add_graph(model.module, dummy_input)  # 实际的模型位于 model.module

        # 确保所有进程在继续前同步
        dist.barrier()

        # 训练循环
        best_accuracy = 0.0
        for epoch in range(1, num_epochs + 1):
            start_epoch_time = time.time()  # 记录每个epoch开始的时间

            train_sampler.set_epoch(epoch)  # 让不同进程加载不同数据
            test_sampler.set_epoch(epoch)  # 让测试数据在不同 epoch 也分布式处理
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # 计算数据大小（MB）
                batch_data_size = images.element_size() * images.nelement() + labels.element_size() * labels.nelement()
                batch_data_size /= 1024 * 1024  # 转换为 MB

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct / total
            print(f"[Rank {rank}] Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # 记录每个epoch的训练时间
            end_epoch_time = time.time()
            epoch_time = end_epoch_time - start_epoch_time

            # 使用 all_reduce 取所有进程中的最大值
            epoch_time_tensor = torch.tensor(epoch_time, dtype=torch.float32, device=device)
            dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)

            # 仅 rank == 0 进程记录最终训练时间
            if rank == 0:
                final_epoch_time = epoch_time_tensor.item()
                writer.add_scalar("Time/Train/distribute", final_epoch_time, epoch)

            # 更新学习率
            scheduler.step()  # 在每个 epoch 结束时调整学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Rank {rank}] Learning rate after epoch {epoch}: {current_lr}")

            # 使用 all_reduce 汇总训练 loss 和 accuracy
            loss_tensor = torch.tensor(epoch_loss, dtype=torch.float32, device=device)
            accuracy_tensor = torch.tensor(epoch_accuracy, dtype=torch.float32, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)

            if rank == 0:
                avg_epoch_loss = loss_tensor.item() / world_size
                avg_epoch_accuracy = accuracy_tensor.item() / world_size
                writer.add_scalar("Loss/train/distribute", avg_epoch_loss, epoch)
                writer.add_scalar("Accuracy/train/distribute", avg_epoch_accuracy, epoch)

            # 评估模型
            model.eval()
            correct_test, total_test = 0, 0
            test_loss_sum = 0.0  # 新增：累计 loss
            with torch.no_grad():  # 关闭反向传播
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    loss = loss_fn(outputs, labels)  # 计算 loss

                    _, predicted = torch.max(outputs, 1)
                    correct_test += (predicted == labels).sum().item()
                    total_test += labels.size(0)

                    # 累计 loss 计数
                    test_loss_sum += loss.item()

            # 使用 all_reduce 汇总测试结果
            correct_test_tensor = torch.tensor(correct_test, dtype=torch.float32, device=device)
            total_test_tensor = torch.tensor(total_test, dtype=torch.float32, device=device)
            test_loss_sum_tensor = torch.tensor(test_loss_sum, dtype=torch.float32, device=device)

            dist.all_reduce(correct_test_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_test_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_loss_sum_tensor, op=dist.ReduceOp.SUM)

            if rank == 0:
                final_test_accuracy = correct_test_tensor.item() / total_test_tensor.item()
                final_test_loss = test_loss_sum_tensor.item() / world_size
                print(f"[Rank 0] Final Test Accuracy after epoch {epoch}: {final_test_accuracy:.4f}")

                writer.add_scalar("Accuracy/test/distribute", final_test_accuracy, epoch)
                writer.add_scalar("Loss/test/distribute", final_test_loss, epoch)

                # 只有 rank 0 负责保存模型
                if final_test_accuracy > best_accuracy:
                    best_accuracy = final_test_accuracy
                    model_path = os.path.join(model_save_path, f'best_model.pth')
                    for file in os.listdir(model_save_path):
                        if file.endswith(".pth"):
                            os.remove(os.path.join(model_save_path, file))  # 如果路径存在，删除该文件夹中的所有.pth文件
                    torch.save(model.state_dict(), model_path)
                    print(f"[Rank {rank}] New best model saved with accuracy: {best_accuracy:.4f}")

            # 同步等待所有进程完成当前超参数训练
            dist.barrier()
        print(f"[Rank {rank}] Finished training for parameter set {param_id}")

        if rank == 0:
            writer.close()

        session.close()  # 释放session

    cleanup()  # 训练结束，清理分布式进程


# ------------------分布式启动入口------------------#
if __name__ == '__main__':
    train()
