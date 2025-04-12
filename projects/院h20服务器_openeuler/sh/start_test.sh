# export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ens19f0np0
sudo systemctl stop firewalld
source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate sgl45
torchrun --nproc-per-node=4 --nnodes=2 --node-rank=0 --master-addr=192.168.88.1 --master-port=28002 test.py