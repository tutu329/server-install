export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate vllm72
./gpu.sh
python -m vllm.entrypoints.openai.api_server --served-model-name=r1-32b --model=/home/tutu/models/DeepSeek-R1-Distill-Qwen-32B --gpu-memory-utilizatio=0.9 --tensor-parallel-size=8 --trust-remote-code --host=0.0.0.0 --port=8001 --max-log-len=1000 --ssl-keyfile=/home/tutu/ssl/powerai.key --ssl-certfile=/home/tutu/ssl/powerai_public.crt --ssl-ca-certs=/home/tutu/ssl/powerai_chain.crt --dtype=half
# --enable-reasoning --reasoning-parser deepseek_r1
