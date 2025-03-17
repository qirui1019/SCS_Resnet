from flask import Flask, jsonify, Response, request, send_file, stream_with_context
import os
import tarfile

app = Flask(__name__)

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


@app.route("/list", methods=["GET"])
def list_files():
    """ 列出数据集文件 """
    return jsonify(os.listdir(EXTRACTED_FOLDER))


@app.route("/datasets/<path:filename>", methods=["GET"])
def download_file(filename):
    """ 以流式方式提供文件 """
    file_path = os.path.join(EXTRACTED_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return Response(stream_with_context(open(file_path, "rb")), content_type="application/octet-stream")


if __name__ == "__main__":
    extract_dataset()
    app.run(host="0.0.0.0", port=5000)

