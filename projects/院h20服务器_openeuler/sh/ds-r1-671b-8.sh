export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGL_ENABLE_JIT_DEEPGEMM=1
source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate sgl45
python -m sglang.launch_server --served-model-name=ds-r1-671b --model-path=/home/tutu/models/DeepSeek-R1 \
--enable-p2p-check --reasoning-parser=deepseek-r1 --trust-remote-code --host=0.0.0.0 --port=28001 \
--mem-fraction-static=0.9 \
--random-seed=446369220 \
--tp=8 --max-total-tokens=65000 --max-running-requests=64 \
# --enable-flashmla \
# --enable-nccl-nvls \
# --enable-deepep-moe \
--disable-radix \
# --attention-backend=fa3 \
# --enable-torch-compile --torch-compile-max-bs 1 \
--enable-deepep-moe \
--deepep-mode=auto \
# --disable-cuda-graph \
--disable-radix-cache \
--enable-ep-moe \
--speculative-algorithm EAGLE3 \
# --speculative-algo NEXTN \
--speculative-draft /home/tutu/models/DeepSeek-R1-NextN \
--speculative-num-steps 2 \
--speculative-eagle-topk 2 \
--speculative-num-draft-tokens 2

