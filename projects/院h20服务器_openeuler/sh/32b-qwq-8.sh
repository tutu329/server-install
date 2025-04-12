export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate vllm82
python -m vllm.entrypoints.openai.api_server --served-model-name=qwq-32b --model=/home/tutu/models/QwQ-32B --gpu-memory-utilizatio=0.5 --tensor-parallel-size=8 --trust-remote-code --host=0.0.0.0 --port=28001 --max-log-len=1000 --max-model-len=15000 --max-num-seqs=8 --enable-reasoning --reasoning-parser=deepseek_r1
# --kv-cache-dtype fp8_e4m3
