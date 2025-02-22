# ======================单节点版本(FP8最多TP16，BF16最多TP32)=====================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate sglxx
./gpu.sh
python -m sglang.launch_server \
--served-model-name=xxx --model-path=/xxx --host=0.0.0.0 --port=8001 --tp=8 --trust-remote-code \
--enable-p2p-check --mem-fraction-static=0.9 \
--max-total-tokens=65536 \
--random-seed 1234 \
--enable-flashinfer-mla \
# 开启nextN(MTP, https://github.com/sgl-project/sglang/pull/3582)
  # 1、Export the weights of nextn layer with script scripts/export_deepseek_nextn.py
  # python export_deepseek_nextn.py --input-dir /path/to/DeepSeek-V3 --output-dir /path/to/DeepSeek-V3-NextN
  # 2、Use the nextn layer as draft model and launch the server
  # python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --speculative-algo NEXTN --speculative-draft /path/to/DeepSeek-V3-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --disable-radix --tp 8
  # 3、运行实际参数
--speculative-algo NEXTN \
--speculative-draft /sgl-workspace/DeepSeek-V3-nextn \
--speculative-num-steps 2 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 4 \
# 开启torch-compile
--disable-radix -enable-torch-compile --torch-compile-max-bs 1
# ============================================================================

# ======================多节点版本(FP8最多TP16，BF16最多TP32)=====================
# 节点0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate sglxx
./gpu.sh
python -m sglang.launch_server \
--served-model-name=xxx --model-path=/xxx \
--enable-p2p-check --mem-fraction-static=0.9 \
--max-total-tokens=65536 \
--random-seed 1234 \
--enable-flashinfer-mla \
--speculative-algo NEXTN \
--speculative-draft /sgl-workspace/DeepSeek-V3-nextn \
--speculative-num-steps 2 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 4 \
--disable-radix -enable-torch-compile --torch-compile-max-bs 1 \
--trust-remote-code \
--tp 16 \
--dist-init-addr FIRST_NODE_IP:5000 \
--nnodes 2 \
--node-rank 0 \
--host 0.0.0.0 --port 8001
# 节点1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate sglxx
./gpu.sh
python -m sglang.launch_server \
--served-model-name=xxx --model-path=/xxx \
--enable-p2p-check --mem-fraction-static=0.9 \
--max-total-tokens=65536 \
--random-seed 1234 \
--enable-flashinfer-mla \
--speculative-algo NEXTN \
--speculative-draft /sgl-workspace/DeepSeek-V3-nextn \
--speculative-num-steps 2 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 4 \
--disable-radix -enable-torch-compile --torch-compile-max-bs 1 \
--trust-remote-code
--tp 16 \
--dist-init-addr FIRST_NODE_IP:5000 \
--nnodes 2 \
--node-rank 1
# ============================================================================

# benchmark
python -m sglang.bench_one_batch_server --model None --base-url https://powerai.cc:8001 --batch-size 1 --input-len 256 --output-len 256


# 1、如果有CUDA graph loading挂起问题
# update nccl to nccl 2.24，fixed hangs when running with different CPU architectures. https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-24-3.html#rel_2-24-3

# 2、如果有watch dog超时问题
# --watchdog-timeout 36000（这个似乎不能解决问题。最新版本应该已经修复了watch dog超时的bug了。）