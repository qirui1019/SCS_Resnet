import pickle
import numpy as np
import torch
# from matplotlib import pyplot as plt
# from scipy.interpolate import make_interp_spline
from torch.utils.data import Dataset


# 读取数据集文件并将其解析成 Python 字典，然后返回该字典对象
def unpickle(file):
    with open(file, 'rb') as fo:  # 以二进制读取的形式打开file,并将文件对象赋值给变量fo
        dict = pickle.load(fo, encoding='bytes')  # 从打开的文件中以字节编码的形式解析数据
    return dict


def load_data(root_path, file_name):
    data_all = unpickle(root_path + file_name)
    # 字典中提取出键为b'data'的图像数据，这些图像数据是以一维数组的形式存储
    images = data_all[b'data']
    # 将图像数据images重新整形为一个四维数组，参数分别是图像数量、高度、宽度、通道数
    images = np.reshape(images, [images.shape[0], 32, 32, 3])
    # 从字典中提取出键为b'labels的值，即标签数据，标签数据是一个包含了每张图像对应类别的列表
    labels = data_all[b'labels']
    return images, labels


# 实现自定义数据集的加载
class FashionDataset(Dataset):
    # 加载图像数据和标签数据
    def __init__(self, mode, root_path, file_name, transform=None):
        """Method to initialize variables."""
        # 如果是训练集，就加载该数据集，并将图像数据和标签数据传入
        if mode == 'train':
            self.images, self.labels = load_data(root_path, file_name)
        # 如果是测试集，加载测试集，并将图像数据和标签数据转入
        if mode == 'test':
            self.images, self.labels = load_data(root_path, file_name)
        # 转换操作
        self.transform = transform  # torch.toTensor()

    # 根据索引返回图像和标签，并执行必要的图像转换
    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 返回数据集的大小
    def __len__(self):
        return len(self.images)


# # 学习率调度器函数，参数分别是优化器对象、当前训练的轮数、初始学习率、学习率衰减周期
# def exp_lr_scheduler(optimizer, epoch, decay_rate, init_lr, lr_decay):
#     # 计算当前伦次下的学习率
#     lr = init_lr * (decay_rate ** (epoch // lr_decay))
#     # 如果当前轮次是学习率衰减周期的整数倍，即到达了一个学习率更新点
#     if epoch % lr_decay == 0:
#         print('LR is set to {}'.format(lr))
#     # 遍历优化器中的参数数组，然后更新学习率
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     # 返回优化器对象
#     return optimizer
#
#
# # 计算模型在测试集或验证集上的性能，返回的是模型的准确率
# def evaluate_model(model, device, data_loader):
#     model.eval()  # 将模型设置为评估模式
#     correct = 0  # 正确预测的数量
#     total = 0  # 总样本数
#     with torch.no_grad():    # 关闭反向传播，因为在评估过程中，不需要进行反向传播或更新权重
#         for images, labels in data_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     return accuracy

