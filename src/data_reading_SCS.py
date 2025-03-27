import torch
from torch.utils.data import Dataset
import requests
import io
from PIL import Image


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

        info_url = f"{STORAGE_SERVER}/datasets/{self.mode}/{index}/info"
        image_url = f"{STORAGE_SERVER}/datasets/{self.mode}/{index}"

        # 获取标签信息
        try:
            info_response = self.session.get(info_url, timeout=10)
            info_response.raise_for_status()  # 确保请求成功
            info = info_response.json()
            label = info.get('label', -1)  # 获取标签，失败时返回 -1

        except requests.RequestException as e:
            raise ValueError(f"获取索引 {index} 信息失败: {e}")

        # 获取图片
        try:
            with self.session.get(image_url, stream=True, timeout=5) as image_response:
                image_response.raise_for_status()
                image_data = image_response.content
                img = Image.open(io.BytesIO(image_data)).convert("RGB")

        except requests.RequestException as e:
            raise ValueError(f"获取索引 {index} 图片失败: {e}")

        # 进行数据增强
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


