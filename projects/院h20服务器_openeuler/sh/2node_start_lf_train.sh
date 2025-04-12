export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# export NCCL_NET_GDR_LEVEL=2       # 建议显式打开 GDR + multi‑rail
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_TC=106
export NCCL_SOCKET_IFNAME=ens19f0np0

source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate lf92
cd ~/LLaMA-Factory

FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.88.1 MASTER_PORT=28011 \
llamafactory-cli train /home/tutu/data/train_32b.yaml