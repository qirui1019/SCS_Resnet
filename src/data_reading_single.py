import pickle
import numpy as np
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



