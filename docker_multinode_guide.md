# Simulating Multi-Node Training with Docker

This guide shows how to run multiple Docker containers on a single machine to mimic a multi-node setup. Each container acts as a worker node and communicates over a user-defined Docker network. With only one GPU available, the containers can share that GPU or run in CPU mode for demonstration purposes.

## 1. Build or Pull a Training Image

1. Start from a CUDA-enabled base image or the framework image provided by this repository.
2. Install the training script and Python requirements inside the image:
   ```Dockerfile
   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   WORKDIR /workspace
   COPY . /workspace
   RUN pip install -r requirements.txt
   ```
3. Build it locally:
   ```bash
   docker build -t qwen-fsdp .
   ```

## 2. Create a Docker Network

All containers must share the same network to communicate over a common master address and port:

```bash
docker network create qwen-net
```

## 3. Launch the Containers

Choose a master port and set the total `WORLD_SIZE` to the number of containers. For two workers on one GPU, run:

```bash
MASTER_ADDR=localhost
MASTER_PORT=29500
WORLD_SIZE=2
```

In separate terminals, launch each container with a distinct `RANK` and the same network:

```bash
# Terminal 1 (rank 0)
docker run --gpus all --rm --network qwen-net \
  -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT \
  -e RANK=0 -e WORLD_SIZE=$WORLD_SIZE \
  qwen-fsdp python finetune_qwen_fsdp.py

# Terminal 2 (rank 1)
docker run --gpus all --rm --network qwen-net \
  -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT \
  -e RANK=1 -e WORLD_SIZE=$WORLD_SIZE \
  qwen-fsdp python finetune_qwen_fsdp.py
```

Both containers will connect to the same master address and coordinate as if running on separate nodes. They can share the GPU or run in CPU mode by omitting `--gpus all`.

## 4. Verify the Run

Check the logs from each container to confirm that the ranks have joined the process group and started training. When finished, clean up the network:

```bash
docker network rm qwen-net
```

This approach allows basic testing of multi-node logic without needing multiple machines.
