export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate vllm73
./gpu.sh
python -m vllm.entrypoints.openai.api_server --served-model-name=qwq-32b --model=/home/tutu/models/QwQ-32B --gpu-memory-utilizatio=0.9 --tensor-parallel-size=8 --trust-remote-code --enable-reasoning --reasoning-parser deepseek_r1 --host=0.0.0.0 --port=8001 --max-log-len=1000 --ssl-keyfile=/home/tutu/ssl/powerai.key --ssl-certfile=/home/tutu/ssl/powerai_public.crt --ssl-ca-certs=/home/tutu/ssl/powerai_chain.crt --dtype=half --max-num-seqs=128
