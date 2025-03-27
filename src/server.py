from flask import Flask, jsonify, request, send_file
import os
import pickle
import io
from PIL import Image
import tarfile
from gevent.pywsgi import WSGIServer
import time
import threading
import logging

# # 设置日志配置
# LOG_FILE = "../data_transfer.log"
# logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")


app = Flask(__name__)


# 定义数据集存储文件夹
DATASET_FOLDER = os.path.join(os.path.abspath(os.sep), "datasets")  # 在计算机根目录下创建
os.makedirs(DATASET_FOLDER, exist_ok=True)  # 确保文件夹存在,无则创建

# 数据集压缩文件
DATASET_ARCHIVE = os.path.join(DATASET_FOLDER, "cifar-10-python.tar.gz")
EXTRACTED_FOLDER = os.path.join(DATASET_FOLDER, "cifar-10-batches-py")

last_request_time = time.time()  # 模拟的数据请求计时器，用于检查请求状态
data_transferred = 0  # 用于记录传输的数据量
last_report_time = time.time()  # 用于监控请求超时
transfer_rate = 0  # 新增：数据传输速率（KB/s）

# 服务器激活标志
server_active = True


def extract_dataset():
    """ 仅当数据集未解压时解压 """
    if not os.path.exists(EXTRACTED_FOLDER):
        print("解压数据集...")
        try:
            with tarfile.open(DATASET_ARCHIVE, "r:gz") as tar:
                tar.extractall(DATASET_FOLDER)
            print("数据集解压完成！")
        except Exception as e:
            print(f"解压失败: {e}")
    else:
        print("数据集已解压，无需重复解压。")


# 缓存 batch 数据，避免重复读取
batch_cache = {}


def load_batch(batch_file):
    """ 加载 CIFAR-10 batch 文件并缓存 """
    if batch_file in batch_cache:
        return batch_cache[batch_file]

    batch_path = os.path.join(EXTRACTED_FOLDER, batch_file)
    if not os.path.exists(batch_path):
        return None

    with open(batch_path, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')

    batch_cache[batch_file] = batch_data
    return batch_data


def get_image_from_batch(batch_file, index):
    """ 从 batch 文件中获取指定索引的图片数据 """
    batch_data = load_batch(batch_file)
    if batch_data is None or index >= len(batch_data[b'data']):
        return None, None

    images = batch_data[b'data']
    labels = batch_data[b'labels']

    image = images[index].reshape(3, 32, 32).transpose(1, 2, 0)
    label = labels[index]

    return image, label


@app.route("/datasets/<mode>/<int:image_id>", methods=["GET"])
def stream_image(mode, image_id):
    """ 根据索引 image_id 返回图片流"""
    global last_request_time
    last_request_time = time.time()  # 更新最后请求时间

    global data_transferred, last_report_time, transfer_rate

    global server_active
    if not server_active:
        # 服务器处于休眠状态，重置标志为激活
        server_active = True

    if mode not in ["train", "test"]:
        return jsonify({"error": "Invalid mode, must be 'train' or 'test'"}), 400

    batch_index = image_id // 10000  # 每个 batch 10000 张图片
    index_in_batch = image_id % 10000
    batch_file = f"data_batch_{batch_index + 1}" if mode == "train" else "test_batch"

    image, label = get_image_from_batch(batch_file, index_in_batch)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    img = Image.fromarray(image)
    img_io = io.BytesIO()
    img.save(img_io, format="JPEG", quality=85)  # 压缩为 JPEG
    img_io.seek(0)

    # 计算图像大小并更新传输数据量
    image_size = len(img_io.getvalue())  # 获取字节数
    data_transferred += image_size

    # 监控数据传输速率，每秒输出一次
    current_time = time.time()
    if current_time - last_report_time >= 1:
        transfer_rate = data_transferred / (current_time - last_report_time) / 1024  # KB/s
        # # 使用 logging 记录数据传输速率
        # logging.info(f"数据传输速率: {transfer_rate:.2f} KB/s")
        # 重置统计
        data_transferred = 0
        last_report_time = current_time

    response = send_file(img_io, mimetype="image/jpeg", as_attachment=False)
    response.headers["Connection"] = "keep-alive"  # 允许持久连接
    return response


@app.route("/datasets/<mode>/<int:image_id>/info", methods=["GET"])
def get_label(mode, image_id):
    """ 根据索引返回标签信息 """
    global last_request_time
    last_request_time = time.time()  # 更新最后请求时间
    global data_transferred, last_report_time, transfer_rate

    if mode not in ["train", "test"]:
        return jsonify({"error": "Invalid mode, must be 'train' or 'test'"}), 40

    batch_index = image_id // 10000
    index_in_batch = image_id % 10000
    batch_file = f"data_batch_{batch_index + 1}" if mode == "train" else "test_batch"

    _, label = get_image_from_batch(batch_file, index_in_batch)
    if label is None:
        return jsonify({"error": "Label not found"}), 404

        # 计算 JSON 响应的大小
    response_data = jsonify({"label": label})
    json_size = len(response_data.get_data())  # 获取 JSON 的字节数
    data_transferred += json_size  # 统计传输量

    # 监控数据传输速率
    current_time = time.time()
    if current_time - last_report_time >= 1:
        transfer_rate = data_transferred / (current_time - last_report_time) / 1024  # KB/s
        data_transferred = 0  # 重置统计
        last_report_time = current_time

    response = response_data
    response.headers["Connection"] = "keep-alive"
    return response


@app.route("/transfer_rate", methods=["GET"])
def get_transfer_rate():
    """ 提供当前数据传输速率，供客户端查询 """
    response = jsonify({"transfer_rate": f"{transfer_rate:.2f} KB/s"})
    response.headers["Connection"] = "keep-alive"
    return response


def check_request_timeout():
    """ 检查是否超过一段时间没有请求，触发日志记录或其他操作 """
    global last_request_time, server_active
    while True:
        time.sleep(60)  # 每 60s 检查一次
        current_time = time.time()
        if current_time - last_request_time > 60 and server_active:
            server_active = False  # 标记服务器为休眠状态


if __name__ == "__main__":
    # 启动超时检查线程
    timeout_thread = threading.Thread(target=check_request_timeout, daemon=True)
    timeout_thread.start()

    # 使用 Gevent 启动 WSGI 服务器
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()


