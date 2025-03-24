# 为了--gpus all正确运行
# 需要:
# 1、sudo apt install -y nvidia-docker2（涉及daemon.json，选择N保留即可）
# 2、sudo systemctl restart docker

cd vllm/examples
sudo bash run_cluster.sh \
                vllm/vllm-openai \
                172.19.80.49 \
                --worker \
                ~/.cache/huggingface/ \
                -e VLLM_HOST_IP=172.19.80.49

#bash run_cluster.sh \
#                vllm/vllm-openai \
#                ip_of_head_node \
#                --worker \
#                /path/to/the/huggingface/home/in/this/node \
#                -e VLLM_HOST_IP=ip_of_this_node