source /home/tutu/anaconda3/etc/profile.d/conda.sh
conda activate client
cd /home/tutu/server/life-agent
python -m streamlit run redis_proxy/redis_monitor.py --server.port 8009