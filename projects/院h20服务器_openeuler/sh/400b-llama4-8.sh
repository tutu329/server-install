export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate vllm
python -m vllm.entrypoints.openai.api_server --served-model-name=llama4-400b --model=/home/tutu/models/Llama-4-Maverick-17B-128E-Instruct-FP8 --gpu-memory-utilizatio=0.9 --tensor-parallel-size=8 --trust-remote-code --host=0.0.0.0 --port=28001 --max-log-len=1000 --max-model-len=128000 --max-num-seqs=64
# --kv-cache-dtype fp8_e4m3
