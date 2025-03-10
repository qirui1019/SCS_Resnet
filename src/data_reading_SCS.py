import pickle
import numpy as np
import requests
import io
import time
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 存储服务器的 URL
STORAGE_SERVER = "http://127.0.0.1:5000"


def unpickle_from_http(file_url, max_retries=3, chunk_size=8192):
    """ 流式下载并解析 pickle 数据 """
    for attempt in range(max_retries):
        try:
            response = requests.get(file_url, stream=True, timeout=10)
            if response.status_code == 200:
                buffer = io.BytesIO()  # 创建流式缓存
                for chunk in response.iter_content(chunk_size=chunk_size):
                    buffer.write(chunk)  # 逐块写入内存缓存
                buffer.seek(0)  # 复位指针，准备读取

                # unpickler = pickle.Unpickler(buffer)
                # data = unpickler.load()  # 逐块解析数据
                # 关键修改：在 Unpickler 初始化时传入 encoding 参数
                unpickler = pickle.Unpickler(buffer, encoding="bytes")  # 指定编码方式 强制使用二进制模式解析
                data = unpickler.load()  # 逐块解析数据
                return data
            else:
                print(f"尝试 {attempt + 1}: 服务器返回错误 {response.status_code}")  # 请求成功，但服务器返回错误
        except requests.RequestException as e:
            print(f"尝试 {attempt + 1}: 请求失败 {e}")  # 请求未发送成功
        time.sleep(2)

    raise ValueError(f"无法从 {file_url} 获取数据，已重试 {max_retries} 次")


def load_data(file_name):
    """ 流式加载数据 """
    file_url = f"{STORAGE_SERVER}/datasets/{file_name}?stream=1"
    data_all = unpickle_from_http(file_url)

    # 解析数据
    # 字典中提取出键为b'data'的图像数据，这些图像数据是以一维数组的形式存储
    images = data_all[b'data']
    # 将图像数据images重新整形为一个四维数组，参数分别是图像数量、高度、宽度、通道数
    images = np.reshape(images, [images.shape[0], 32, 32, 3])
    # 从字典中提取出键为b'labels的值，即标签数据，标签数据是一个包含了每张图像对应类别的列表
    labels = data_all[b'labels']

    return images, labels


class FashionDataset(Dataset):
    """ 自定义 PyTorch 数据集，支持流式读取 """

    def __init__(self, mode, file_name, transform=None):
        if mode not in ['train', 'test']:
            raise ValueError("mode 只能是 'train' 或 'test'")

        # self.file_name = file_name  # 仅保存文件名，避免提前加载
        # self.transform = transform  # 预处理操作

        self.images, self.labels = load_data(file_name)  # 初始化时加载一次
        self.transform = transform if transform else transforms.ToTensor() # 预处理操作

    def __getitem__(self, index):
        # images, labels = load_data(self.file_name)  # 每次读取时从服务器加载
        # image, label = images[index], labels[index]
        #
        # if self.transform is not None:
        #     image = self.transform(image)
        # return image, label
        image, label = self.images[index], self.labels[index]

        if self.transform:
            image = self.transform(image)  # 应用数据增强/转换

        # return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return image.clone().detach().float(), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # images, _ = load_data(self.file_name)  # 获取数据集大小
        # return len(images)

        return len(self.images)  # 直接返回长度，不用每次请求服务器




