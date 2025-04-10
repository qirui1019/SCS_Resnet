# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import threading

from train_SCS import train_resnet

app = Flask(__name__)
CORS(app)

# Training state
training_state = {
    "is_training": False,
    "epoch": 0,
    "max_epoch": 10
}

# TensorBoard writer
# writer = SummaryWriter()


@app.route('/start', methods=['POST'])
def start_training():
    print('start_training')
    data = request.json
    num_epoch = data.get('epochs', 10)
    mode = data.get('mode', 'local_train')
    if not training_state["is_training"]:
        training_state["is_training"] = True
        training_state["epoch"] = 0
        training_state["max_epoch"] = num_epoch  # 动态更新

        if mode == 'local_train':
            thread = threading.Thread(target=train_resnet, args=(num_epoch, training_state))
            thread.start()
        elif mode == 'distributed_train':
            master_addr = data.get('master_addr', '10.27.251.68')
            master_port = str(data.get('master_port', '29500'))
            nproc_per_node = str(data.get('nproc_per_node', 2))
            nnodes = str(data.get('nnodes', 1))
            cmd = [
                'torchrun',
                f'--nproc_per_node={nproc_per_node}',
                f'--nnodes={nnodes}',
                f'--master_addr={master_addr}',
                f'--master_port={master_port}',
                'train_distribute.py',
                f'--epochs={num_epoch}'
            ]

            print("Launching subprocess:", ' '.join(cmd))
            subprocess.Popen(cmd)  # 非阻塞方式启动
        else:
            return jsonify({"status": "Invalid mode"}), 400

        return jsonify({"status": "Training started"}), 200
    else:
        return jsonify({"status": "Training already in progress"}), 400


@app.route('/stop', methods=['POST'])
def stop_training():
    print('stop_training')
    training_state["is_training"] = False
    return jsonify({"status": "Training stopped"}), 200


@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(training_state), 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)
