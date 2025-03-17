import os
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_reading_SCS import FashionDataset
import model_ResNet18 as Rn


class Trainer:
    def __init__(self, config):
        self.config = config  # 存储超参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error = nn.CrossEntropyLoss()  # 交叉熵损失

        # 处理模型存储路径
        self.model_save_path = config["model_save_path"]
        # 检查模型保存路径是否存在
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)  # 如果模型保存路径不存在，则创建

        # 读取数据
        self.train_loader, self.test_loader = self.load_data()

    def load_data(self):
        """加载训练集和测试集"""
        # 创建一个FashionDataset变量，表示类型是训练集/测试集，并转换为Pytorch的张量格式
        # 使用DataLoader类将上一步创建的训练集train_set封装成一个批量生成器，每次从数据集中加载 batch_size 个样本
        # 定义数据增强
        transform = transforms.Compose([
            transforms.ToTensor(),  # 数据预处理
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

        # 创建数据集对象
        train_set = FashionDataset(mode="train", file_name=file_name1, transform=transform)
        # 使用 DataLoader 进行批量加载
        train_loader = DataLoader(
            train_set,
            batch_size=256,  # 每次训练加载 256 张
            shuffle=True,  # 每个 epoch 重新打乱数据,提高模型泛化能力
            num_workers=3,  # 使用 3个 CPU 线程加载数据
            pin_memory=False,  # 如果使用 GPU，建议设为 True
            prefetch_factor=2  # 预取数据，加快加载
        )

        # 创建数据集对象
        test_set = FashionDataset(mode="test", file_name=file_name2, transform=transform)
        # 使用 DataLoader 进行批量加载
        test_loader = DataLoader(
            test_set,
            batch_size=256,  # 每次训练加载 256 张
            shuffle=False,  # 每个 epoch 重新打乱数据,提高模型泛化能力
            num_workers=3,  # 使用 3个 CPU 线程加载数据
            pin_memory=False,  # 如果使用 GPU，建议设为 True
            prefetch_factor=2  # 预取数据，加快加载
        )

        return train_loader, test_loader

    def evaluate_model(self, model):
        """在测试集上评估模型准确率"""
        model.eval()  # 将模型设置为评估模式
        correct = 0  # 正确预测的数量
        total = 0  # 总样本数
        with torch.no_grad():  # 关闭反向传播，因为在评估过程中，不需要进行反向传播或更新权重
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def exp_lr_scheduler(self, optimizer, epoch, initial_lr, lr_decay, decay_rate):
        """学习率调度器"""
        if lr_decay:
            lr = initial_lr * (decay_rate ** (epoch // lr_decay))  # 计算当前伦次下的学习率
            # 遍历优化器中的参数数组，然后更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # 如果当前轮次是学习率衰减周期的整数倍，即到达了一个学习率更新点
            if epoch % lr_decay == 0:
                print(f"LR adjusted to {lr}")
        return optimizer

    def train(self, learning_rate, lr_decay, decay_rate, num_epochs):
        """执行模型训练"""
        model = Rn.ResNet18().to(self.device)
        # 定义一个Adam优化器，用于更新模型参数以最小化损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_accuracy = 0

        for epoch in range(1, num_epochs + 1):
            model.train()  # 将模型设置为训练模式
            total_hits = 0  # 统计当前 epoch 中所有批次的正确预测（hit）总数
            total_samples = 0  # 统计当前 epoch 中所有批次的总样本数
            total_loss = 0.0  # 累计当前 epoch 中所有批次的总损失值

            # 根据当前epoch来调整学习率
            optimizer = self.exp_lr_scheduler(optimizer, epoch, learning_rate, lr_decay, decay_rate)

            # 每个epoch重新加载训练数据
            self.train_loader, _ = self.load_data()  # 重新加载数据集，确保每个epoch使用新的数据

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device).long()
                # 优化器梯度清零
                optimizer.zero_grad()

                # 得到模型的预测输出
                outputs = model(images)
                # 计算损失值
                loss = self.error(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_hits += (torch.max(outputs, 1)[1] == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = total_loss / total_samples  # 计算每个epoch的损失
            epoch_accuracy = total_hits / total_samples  # 计算每个epoch的平均精度
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")

            test_accuracy = self.evaluate_model(model)  # 测试集上的精度
            print(f"Test accuracy after epoch {epoch}: {test_accuracy:.4f}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_path = os.path.join(self.model_save_path,
                                               f"batch_size{batch_size}_lr{learning_rate}_lr_decay{lr_decay}_dr{decay_rate}_epoch{epoch}.pth")
                for file in os.listdir(model_save_path):
                    if file.endswith(".pth"):
                        os.remove(os.path.join(model_save_path, file))  # 如果路径存在，删除该文件夹中的所有.pth文件
                torch.save(model.state_dict(), best_model_path)  # 保存模型结构和参数


if __name__ == "__main__":
    batch_size = 256
    file_name1 = "data_batch_1"
    file_name2 = "test_batch"
    model_save_path = "../best_model_ResNet18_SCS"

    learning_rates = [0.001]
    lr_decays = [15, 10]
    decay_rates = [0.5]
    num_epochs_list = [50]

    param_combinations = itertools.product(learning_rates, lr_decays, decay_rates, num_epochs_list)

    for learning_rate, lr_decay, decay_rate, num_epochs in param_combinations:
        print(f"Training with batch_size={batch_size}, lr={learning_rate}, lr_decay={lr_decay}, dr={decay_rate}, epochs={num_epochs}")
        config = {
            "batch_size": batch_size,
            "file_name1": file_name1,
            "file_name2": file_name2,
            "model_save_path": model_save_path
        }
        trainer = Trainer(config)
        trainer.train(learning_rate, lr_decay, decay_rate, num_epochs)
