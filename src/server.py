from flask import Flask, jsonify, Response, request, send_file, stream_with_context
import os
import tarfile

app = Flask(__name__)

# 定义数据集存储文件夹
DATASET_FOLDER = os.path.join(os.path.abspath(os.sep), "datasets")  # 在计算机根目录下创建
os.makedirs(DATASET_FOLDER, exist_ok=True)  # 确保文件夹存在,无则创建

# 数据集压缩包路径
DATASET_ARCHIVE = os.path.join(DATASET_FOLDER, "cifar-10-python.tar.gz")
EXTRACTED_FOLDER = os.path.join(DATASET_FOLDER, "cifar-10-batches-py")  # 解压后的文件夹


def extract_dataset():
    """ 检查数据集是否已经解压，如果未解压，则进行解压 """
    if not os.path.exists(EXTRACTED_FOLDER):  # 如果解压文件夹不存在
        print("正在解压数据集...")
        try:
            with tarfile.open(DATASET_ARCHIVE, "r:gz") as tar:
                tar.extractall(DATASET_FOLDER)  # 解压到数据集文件夹
            print("数据集解压完成！")
        except Exception as e:
            print(f"解压数据集失败: {e}")
    else:
        print("数据集已解压，无需重复解压。")


@app.route("/list", methods=["GET"])
def list_files():
    """列出数据集文件"""
    files = os.listdir(DATASET_FOLDER)
    return jsonify(files)


def generate_large_file(file_path, chunk_size=8192):
    """ 以流的方式读取大文件 """
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk  # 逐块传输数据


@app.route("/datasets/<path:filename>", methods=["GET"])
def download_file(filename):
    """ 通过 HTTP 以流式方式提供文件访问 """
    print(f"Received request for file: {filename}")  # 打印日志，看看是否有请求到达
    file_path = os.path.join(EXTRACTED_FOLDER, filename)
    # print(f"Full path to requested file: {file_path}")  # 检查拼接后的路径是否正确
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # 方式 1：直接使用 send_file，适用于小文件
    if request.args.get("stream") != "1":  # 允许用户选择流式方式
        return send_file(file_path, as_attachment=False)

    # 方式 2：流式传输大文件
    return Response(stream_with_context(generate_large_file(file_path)), content_type="application/octet-stream")


if __name__ == "__main__":
    extract_dataset()  # 服务器启动时解压数据集
    # print("Flask server started! Listening on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
