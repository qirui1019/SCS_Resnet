from flask import Flask, jsonify, request, send_file
import os
import pickle
import io
from PIL import Image
import tarfile
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 定义数据集存储文件夹
DATASET_FOLDER = os.path.join(os.path.abspath(os.sep), "datasets")  # 在计算机根目录下创建
os.makedirs(DATASET_FOLDER, exist_ok=True)  # 确保文件夹存在,无则创建

# 数据集压缩文件
DATASET_ARCHIVE = os.path.join(DATASET_FOLDER, "cifar-10-python.tar.gz")
EXTRACTED_FOLDER = os.path.join(DATASET_FOLDER, "cifar-10-batches-py")


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

    response = send_file(img_io, mimetype="image/jpeg", as_attachment=False)
    response.headers["Connection"] = "keep-alive"  # 允许持久连接
    return response


@app.route("/datasets/<mode>/<int:image_id>/info", methods=["GET"])
def get_label(mode, image_id):
    """ 根据索引返回标签信息 """
    if mode not in ["train", "test"]:
        return jsonify({"error": "Invalid mode, must be 'train' or 'test'"}), 40

    batch_index = image_id // 10000
    index_in_batch = image_id % 10000
    batch_file = f"data_batch_{batch_index + 1}" if mode == "train" else "test_batch"

    _, label = get_image_from_batch(batch_file, index_in_batch)
    if label is None:
        return jsonify({"error": "Label not found"}), 404

    response = jsonify({"label": label})
    response.headers["Connection"] = "keep-alive"
    return response


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, threaded=True)
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()  # 使用 Gevent 启动 WSGI 服务器



# from flask import Flask, jsonify, Response, request, send_file, stream_with_context
# import os
# import tarfile
#
# app = Flask(__name__)
#
# # 定义数据集存储文件夹
# DATASET_FOLDER = os.path.join(os.path.abspath(os.sep), "datasets")  # 在计算机根目录下创建
# os.makedirs(DATASET_FOLDER, exist_ok=True)  # 确保文件夹存在,无则创建
#
# # 数据集压缩文件
# DATASET_ARCHIVE = os.path.join(DATASET_FOLDER, "cifar-10-python.tar.gz")
# EXTRACTED_FOLDER = os.path.join(DATASET_FOLDER, "cifar-10-batches-py")
#
#
# def extract_dataset():
#     """ 仅当数据集未解压时解压 """
#     if not os.path.exists(EXTRACTED_FOLDER):
#         print("解压数据集...")
#         try:
#             with tarfile.open(DATASET_ARCHIVE, "r:gz") as tar:
#                 tar.extractall(DATASET_FOLDER)
#             print("数据集解压完成！")
#         except Exception as e:
#             print(f"解压失败: {e}")
#     else:
#         print("数据集已解压，无需重复解压。")
#
#
# @app.route("/list", methods=["GET"])
# def list_files():
#     """ 列出数据集文件 """
#     return jsonify(os.listdir(EXTRACTED_FOLDER))
#
#
# @app.route("/datasets/<path:filename>", methods=["GET"])
# def download_file(filename):
#     """ 以流式方式提供文件 """
#     file_path = os.path.join(EXTRACTED_FOLDER, filename)
#     if not os.path.exists(file_path):
#         return jsonify({"error": "File not found"}), 404
#
#     return Response(stream_with_context(open(file_path, "rb")), content_type="application/octet-stream")
#
#
# if __name__ == "__main__":
#     extract_dataset()
#     app.run(host="0.0.0.0", port=5000)
#
