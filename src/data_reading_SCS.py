import torch
from torch.utils.data import Dataset
import requests
import io
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 服务器地址（修改为实际 IP）
STORAGE_SERVER = "http://10.27.251.68:5000"


class FashionDataset(Dataset):
    """ 通过 HTTP 从存储服务器获取数据，使用 requests.Session 复用连接并配置连接池 """

    def __init__(self, mode, total_samples=50000, transform=None, session=None):
        assert mode in ["train", "test"], "mode 只能是 'train' 或 'test'"
        self.mode = mode
        self.total_samples = total_samples
        self.transform = transform
        self.session = session or requests.Session()  # 传入 session，否则新建

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        """ 按索引获取图片和标签 """

        # 获取标签信息
        info_url = f"{STORAGE_SERVER}/datasets/{self.mode}/{index}/info"
        image_url = f"{STORAGE_SERVER}/datasets/{self.mode}/{index}"

        try:
            info_response = self.session.get(info_url, timeout=5)
            info_response.raise_for_status()  # 确保请求成功
            info = info_response.json()
            label = info.get('label', -1)  # 获取标签，失败时返回 -1
        except requests.RequestException as e:
            raise ValueError(f"获取索引 {index} 信息失败: {e}")

        # 获取图片
        try:
            with self.session.get(image_url, stream=True, timeout=5) as image_response:
                image_response.raise_for_status()
                img = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        except requests.RequestException as e:
            raise ValueError(f"获取索引 {index} 图片失败: {e}")

        # 进行数据增强
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        """ 关闭 Session，释放资源 """
        self.session.close()


# import pickle
# import numpy as np
# import requests
# import io
# import time
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
#
# # 存储服务器的 URL
# # STORAGE_SERVER = "http://127.0.0.1:5000"
# STORAGE_SERVER = "http://10.27.251.68:5000"  # 修改为主机的实际 IP
#
#
# def unpickle_from_http(file_url, max_retries=3, chunk_size=8192):
#     """ 流式下载并解析 pickle 数据 """
#     for attempt in range(max_retries):
#         try:
#             response = requests.get(file_url, stream=True, timeout=10)
#             if response.status_code == 200:
#                 buffer = io.BytesIO()  # 创建流式缓存
#                 for chunk in response.iter_content(chunk_size=chunk_size):
#                     buffer.write(chunk)  # 逐块写入内存缓存
#                 buffer.seek(0)  # 复位指针，准备读取
#
#                 # unpickler = pickle.Unpickler(buffer)
#                 # data = unpickler.load()  # 逐块解析数据
#                 # 关键修改：在 Unpickler 初始化时传入 encoding 参数
#                 unpickler = pickle.Unpickler(buffer, encoding="bytes")  # 指定编码方式 强制使用二进制模式解析
#                 data = unpickler.load()  # 逐块解析数据
#                 return data
#             else:
#                 print(f"尝试 {attempt + 1}: 服务器返回错误 {response.status_code}")  # 请求成功，但服务器返回错误
#         except requests.RequestException as e:
#             print(f"尝试 {attempt + 1}: 请求失败 {e}")  # 请求未发送成功
#         time.sleep(2)
#
#     raise ValueError(f"无法从 {file_url} 获取数据，已重试 {max_retries} 次")
#
#
# def load_data(file_name):
#     """ 流式加载数据 """
#     file_url = f"{STORAGE_SERVER}/datasets/{file_name}?stream=1"
#     data_all = unpickle_from_http(file_url)
#
#     # 解析数据
#     # 字典中提取出键为b'data'的图像数据，这些图像数据是以一维数组的形式存储
#     images = data_all[b'data']
#     # 将图像数据images重新整形为一个四维数组，参数分别是图像数量、高度、宽度、通道数
#     images = np.reshape(images, [images.shape[0], 32, 32, 3])
#     # 从字典中提取出键为b'labels的值，即标签数据，标签数据是一个包含了每张图像对应类别的列表
#     labels = data_all[b'labels']
#
#     return images, labels
#
#
# class FashionDataset(Dataset):
#     """ 自定义 PyTorch 数据集，支持流式读取 """
#
#     def __init__(self, mode, file_name, transform=None):
#         if mode not in ['train', 'test']:
#             raise ValueError("mode 只能是 'train' 或 'test'")
#
#         # self.file_name = file_name  # 仅保存文件名，避免提前加载
#         # self.transform = transform  # 预处理操作
#
#         self.images, self.labels = load_data(file_name)  # 初始化时加载一次
#         self.transform = transform if transform else transforms.ToTensor()  # 预处理操作
#
#     def __getitem__(self, index):
#         # images, labels = load_data(self.file_name)  # 每次读取时从服务器加载
#         # image, label = images[index], labels[index]
#         #
#         # if self.transform is not None:
#         #     image = self.transform(image)
#         # return image, label
#         image, label = self.images[index], self.labels[index]
#
#         if self.transform:
#             image = self.transform(image)  # 应用数据增强/转换
#
#         # return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
#         return image.clone().detach().float(), torch.tensor(label, dtype=torch.long)
#
#     def __len__(self):
#         # images, _ = load_data(self.file_name)  # 获取数据集大小
#         # return len(images)
#
#         return len(self.images)  # 直接返回长度，不用每次请求服务器
#
#
#
