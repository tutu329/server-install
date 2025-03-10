export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate sgl41
./gpu.sh
python -m sglang.launch_server --served-model-name=qwen72 --model-path=/home/tutu/models/Qwen2.5-72B-Instruct-GPTQ-Int4 --host=0.0.0.0 --port=8001 --tp=8 --trust-remote-code --max-total-tokens=15000 --enable-p2p-check --mem-fraction-static=0.9 --enable-torch-compile