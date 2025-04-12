export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGL_ENABLE_JIT_DEEPGEMM=1
source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate sgl44
python -m sglang.launch_server --served-model-name=ds-r1-671b --model-path=/home/tutu/models/DeepSeek-R1-Zero --mem-fraction-static=0.9 \
--enable-p2p-check \
--tp=8 --max-total-tokens=65000 --max-running-requests=64 \
--trust-remote-code --host=0.0.0.0 --port=28001 \
--reasoning-parser=deepseek-r1 \
# --enable-torch-compile --torch-compile-max-bs 1 \
--enable-ep-moe \
--speculative-algo NEXTN \
--speculative-draft /home/tutu/models/DeepSeek-R1-Zero-NextN \
--speculative-num-steps 2 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 4

