export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGL_ENABLE_JIT_DEEPGEMM=1
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3

# export NCCL_TOPO_DUMP_FILE=nccl-topo.xml
# export NCCL_TOPO_FILE=nccl-topo-opt.xml

# export NCCL_IB_MAX_QP=128        # 增加 NCCL 队列深度
# export NCCL_IB_TC=106            # 设置 InfiniBand 通信优先级
# export NCCL_SHM_DISABLE=0        # 启用共享内存
# export NCCL_P2P_DISABLE=0        # 启用 GPU 之间的 P2P 通信
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=0
# export NCCL_IB_TC=106

export NCCL_SOCKET_IFNAME=ens19f0np0
export GLOO_SOCKET_IFNAME=ens19f0np0

# # 启用NCCL调试输出
# export NCCL_DEBUG=INFO
# # 启用Gloo调试输出（PyTorch）
# export GLOO_DEBUG=1

source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate sgl45
python -m sglang.launch_server --served-model-name=ds-r1-671b --model-path=/home/tutu/models/DeepSeek-R1 \
--enable-p2p-check \
--reasoning-parser=deepseek-r1 --trust-remote-code \
--tp 16 \
--dist-init-addr 192.168.88.1:28002 \
--nnodes 2 \
--node-rank 0 \
--host=0.0.0.0 --port=28001 \
--mem-fraction-static=0.7 \
--random-seed=446369220 \
--max-total-tokens=65000 --max-running-requests=64 \
# --enable-dp-attention \
# --quantization fp8 \
# --enable-flashmla \
# --enable-nccl-nvls \
# --enable-deepep-moe \
# --deepep-mode=auto \
# --disable-cuda-graph \
# --disable-radix-cache \
# --cuda-graph-max-bs=128 \
# --attention-backend=fa3 \
--enable-torch-compile --torch-compile-max-bs 1 \
--enable-ep-moe \
--speculative-algorithm EAGLE3 \
# --speculative-algo NEXTN \
--speculative-draft /home/tutu/models/DeepSeek-R1-NextN \
--speculative-num-steps 2 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 4

