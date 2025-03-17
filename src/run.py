import subprocess

command = [
    "torchrun",
    "--nnodes=1",
    "--nproc-per-node=2",
    "--master-addr=10.27.251.68",
    "--master-port=29500",
    "train_distribute.py"
]

subprocess.run(command)
