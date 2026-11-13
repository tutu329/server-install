#!/usr/bin/env bash
# Linux宿主机运行；与5_verl_api.py放在同一目录，只输入三个位置参数。
# bash 5_verl_api.sh 0.0.0.0 7877 /workspace/models/Qwen3-8B
# 第三个参数也可以是宿主机路径/home/zs/models/Qwen3-8B：优先复用挂载，否则复制。
# v4：仅使用Docker原生网络；没有自写TCP转发，不使用nsenter，不创建转发容器。
# host网络：第二个参数直接决定监听端口；改变IP/port会优雅重启API（不重启容器）。
# bridge网络：先创建-p IP:PORT:PORT映射；缺少映射则明确报错，不暗中创建转发。
# 本脚本不重建训练容器。host模式的一次性迁移应在训练任务结束后进行。
# 不安装宿主机Python，不执行uv sync，不升级torch/CUDA。
# 默认数据目录/workspace/verl_api_data，应确保/workspace是宿主机挂载目录。
set -Eeuo pipefail
CONTAINER="verl-l40s"
VERL_DIR="/workspace/verl"
REMOTE_DIR="/workspace/verl_api"
STATE_DIR="/workspace/verl_api_data"
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    printf '用法: bash %s [服务IP=0.0.0.0] [端口=7877] [模型path=/workspace/models/Qwen3-8B]\n' "$0"
    exit 0
fi
[[ $# -le 3 ]] || { echo "只接受服务IP、port、模型path三个参数。" >&2; exit 2; }
SERVICE_IP="${1:-0.0.0.0}"
PORT="${2:-7877}"
MODEL="${3:-/workspace/models/Qwen3-8B}"
if [[ ! "$PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
    echo "端口需1..65535。" >&2; exit 2;
fi
PORT="$((10#$PORT))"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$HERE/5_verl_api.py" ]] || { echo "请把两个文件放在同一目录。" >&2; exit 2; }
command -v docker >/dev/null || { echo "宿主机没有docker。" >&2; exit 2; }
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null || { echo "需要Docker权限或sudo。" >&2; exit 2; }
    DOCKER=(sudo docker)
    "${DOCKER[@]}" info >/dev/null
fi
"${DOCKER[@]}" inspect "$CONTAINER" >/dev/null 2>&1 || {
    echo "未找到$CONTAINER；如容器已改名，请修改脚本顶部CONTAINER常量。" >&2; exit 2;
}
if [[ $("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$CONTAINER") != true ]]; then
    "${DOCKER[@]}" start "$CONTAINER" >/dev/null
fi
MODE="$("${DOCKER[@]}" inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER")"
if [[ "$MODE" != host ]]; then
    PUBLISHED="$("${DOCKER[@]}" port "$CONTAINER" "$PORT/tcp" 2>/dev/null || true)"
    MAPPING_OK=no
    while IFS= read -r binding; do
        # 只接受所请求的绑定，避免请求127.0.0.1却意外暴露到所有网卡。
        [[ "$binding" != "$SERVICE_IP:$PORT" ]] || MAPPING_OK=yes
    done <<< "$PUBLISHED"
    if [[ "$MAPPING_OK" != yes ]]; then
        echo "容器网络=$MODE，尚无所需映射 $SERVICE_IP:$PORT -> $PORT/tcp。" >&2
        echo "请先迁移为--network host，或按 -p $SERVICE_IP:$PORT:$PORT 重建容器。" >&2
        echo "host模式以后只需改脚本第二个参数；bridge模式换映射需重建容器。" >&2
        echo "本脚本不创建TCP转发，也不擅自停止/重建训练容器。" >&2
        exit 2
    fi
fi
PY="$VERL_DIR/.venv/bin/python"
"${DOCKER[@]}" exec "$CONTAINER" test -x "$PY" || { echo "请先完成$PY所在的verl环境安装。" >&2; exit 2; }
"${DOCKER[@]}" exec "$CONTAINER" "$PY" -c 'import sys,ipaddress,verl,torch; ipaddress.IPv4Address(sys.argv[1])' "$SERVICE_IP"
if ! "${DOCKER[@]}" exec "$CONTAINER" test -f "$MODEL/config.json"; then
    [[ -d "$MODEL" && -f "$MODEL/config.json" ]] || { echo "宿主机和容器均未找到有效模型: $MODEL" >&2; exit 2; }
    MODEL="$(readlink -f -- "$MODEL")"
    MAPPED="$("${DOCKER[@]}" inspect "$CONTAINER" | "${DOCKER[@]}" exec -i "$CONTAINER" "$PY" -c '
import json,sys
source=sys.argv[1]; choices=[]
for mount in json.load(sys.stdin)[0].get("Mounts",[]):
    prefix=mount["Source"].rstrip("/")
    if source==prefix or source.startswith(prefix+"/"):
        choices.append((len(prefix),mount["Destination"].rstrip("/")+source[len(prefix):]))
print(max(choices)[1] if choices else "")
' "$MODEL")"
    if [[ -n "$MAPPED" ]] && "${DOCKER[@]}" exec "$CONTAINER" test -f "$MAPPED/config.json"; then
        MODEL="$MAPPED"
    else
        SIGNATURE="$(find -L "$MODEL" -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
        DEST="/workspace/verl_api_models/$(basename -- "$MODEL")-${SIGNATURE:0:12}"
        if ! "${DOCKER[@]}" exec "$CONTAINER" test -f "$DEST/.copy-complete-$SIGNATURE"; then
            echo "复制模型到容器专属目录$DEST；大模型可能需要较长时间和磁盘空间..."
            "${DOCKER[@]}" exec "$CONTAINER" mkdir -p "$DEST"
            "${DOCKER[@]}" cp -L "$MODEL/." "$CONTAINER:$DEST/"
            "${DOCKER[@]}" exec "$CONTAINER" touch "$DEST/.copy-complete-$SIGNATURE"
        fi
        MODEL="$DEST"
    fi
fi
"${DOCKER[@]}" exec "$CONTAINER" mkdir -p "$REMOTE_DIR" "$STATE_DIR"
if ! "${DOCKER[@]}" exec "$CONTAINER" "$PY" -c 'import fastapi,uvicorn,httpx,multipart,pydantic; assert int(pydantic.__version__.split(".")[0])>=2' >/dev/null 2>&1; then
    echo "在容器现有虚拟环境中补齐Web接口依赖..."
    DEPS=('fastapi>=0.110,<1' 'uvicorn>=0.27' 'httpx>=0.27' 'python-multipart>=0.0.9' 'pydantic>=2,<3')
    if "${DOCKER[@]}" exec "$CONTAINER" sh -c 'command -v uv >/dev/null'; then
        "${DOCKER[@]}" exec "$CONTAINER" uv pip install --python "$PY" "${DEPS[@]}"
    else
        "${DOCKER[@]}" exec "$CONTAINER" "$PY" -m pip install "${DEPS[@]}"
    fi
fi
"${DOCKER[@]}" exec "$CONTAINER" "$PY" -c 'import pandas,pyarrow,transformers,peft; print("数据处理和LoRA依赖已就绪")'
INNER_IP="0.0.0.0"
[[ "$MODE" != host ]] || INNER_IP="$SERVICE_IP"
CHECK_IP="127.0.0.1"
[[ "$INNER_IP" == 0.0.0.0 ]] || CHECK_IP="$INNER_IP"
HEALTH_CODE='import json,sys,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); assert json.load(o.open("http://"+sys.argv[1]+":"+sys.argv[2]+"/healthz",timeout=2))["service"]=="verl-api"'
if "${DOCKER[@]}" exec "$CONTAINER" "$PY" -c "$HEALTH_CODE" "$CHECK_IP" "$PORT" >/dev/null 2>&1; then
    "${DOCKER[@]}" exec "$CONTAINER" "$PY" -c '
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])/"server_process.json"
if not p.exists(): raise SystemExit("已有API缺少启动记录，请检查现有进程")
if json.loads(p.read_text())["model_path"]!=str(pathlib.Path(sys.argv[2]).resolve()):
    raise SystemExit("现有API起点模型不同；其他模型请通过/models/register登记")
' "$STATE_DIR" "$MODEL"
    echo "API已经运行，复用现有服务；端口=$PORT，网络=$MODE。"
else
    # 已有worker可能继续读取同一源码，不在此情况下覆盖不同版本的文件。
    LOCAL_HASH="$(sha256sum "$HERE/5_verl_api.py" | cut -d' ' -f1)"
    "${DOCKER[@]}" exec -i "$CONTAINER" "$PY" - "$STATE_DIR" "$REMOTE_DIR/5_verl_api.py" "$LOCAL_HASH" <<'PYCHECKCODE'
import hashlib
from pathlib import Path
import sqlite3
import sys
root, path, expected = sys.argv[1:]
source = Path(path)
if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() == expected:
    sys.exit(0)
database = Path(root) / 'state.sqlite3'
if database.exists():
    with sqlite3.connect('file:%s?mode=ro' % database, uri=True) as db:
        active = db.execute("SELECT id,state FROM objects WHERE kind='job' AND state IN ('queued','starting','running','loading','ready','stopping')").fetchall()
    if active:
        sys.exit('存在未结束任务，不能覆盖不同版本的API源码；请使用原版本py，或在任务结束后更新。')
PYCHECKCODE
    # 同一数据目录只允许一个API；在host模式变更监听端口时，优雅关闭旧API。
    # PID和创建时间都核对，避免误杀被系统复用的PID；不停止独立训练worker。
    "${DOCKER[@]}" exec -i "$CONTAINER" "$PY" - "$STATE_DIR" "$REMOTE_DIR/5_verl_api.py" "$INNER_IP" "$PORT" "$MODEL" <<'PYRESTART'
import json
import os
from pathlib import Path
import signal
import sys
import time

root, script, bind, port, model = sys.argv[1:]
record = Path(root) / 'server_process.json'
if not record.exists():
    sys.exit(0)
info = json.loads(record.read_text())
if info.get('model_path') != str(Path(model).resolve()):
    sys.exit('此数据目录绑定的起点模型不同；其他模型请使用/models/register添加。')

def start_tick(pid):
    try:
        text = Path('/proc/%s/stat' % pid).read_text()
        fields = text[text.rfind(')') + 2:].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None

pid = info.get('pid')
if not info.get('pid_start') or start_tick(pid) != info['pid_start']:
    sys.exit(0)
args = Path('/proc/%s/cmdline' % pid).read_bytes().decode().split('\0')
try:
    index = next(i for i, value in enumerate(args) if value and str(Path(value).resolve()) == str(Path(script).resolve()))
    old_bind, old_port = args[index + 1:index + 3]
except (ValueError, StopIteration):
    sys.exit('已有进程的启动参数无法确认，未终止该进程。')
if (old_bind, old_port) == (bind, port):
    sys.exit('同地址API进程仍在运行，但健康检查未通过，请查看server.log后重试。')
print('API监听地址从%s:%s改为%s:%s，正在优雅重启API...' % (old_bind, old_port, bind, port), flush=True)
os.kill(pid, signal.SIGTERM)
for _ in range(120):
    if start_tick(pid) != info['pid_start']:
        sys.exit(0)
    time.sleep(.25)
sys.exit('旧API在30秒内未退出，可能仍有进行中的对话；未强制终止，请稍后重试。')
PYRESTART
    # 相同代码不重复覆盖，避免独立worker读取期间遇到短暂的不完整文件。
    REMOTE_HASH="$("${DOCKER[@]}" exec "$CONTAINER" "$PY" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing")' "$REMOTE_DIR/5_verl_api.py")"
    if [[ "$REMOTE_HASH" != "$LOCAL_HASH" ]]; then
        "${DOCKER[@]}" cp "$HERE/5_verl_api.py" "$CONTAINER:$REMOTE_DIR/5_verl_api.py"
    fi
    "${DOCKER[@]}" exec "$CONTAINER" "$PY" -m py_compile "$REMOTE_DIR/5_verl_api.py"
    # shellcheck disable=SC2016 # 参数刻意在容器内bash展开。
    "${DOCKER[@]}" exec -d -w "$VERL_DIR" "$CONTAINER" bash -c '
ulimit -s 65536 || true
exec "$1" -u "$2" "$3" "$4" "$5" >> "$6" 2>&1
' bash "$PY" "$REMOTE_DIR/5_verl_api.py" "$INNER_IP" "$PORT" "$MODEL" "$STATE_DIR/server.log"
    READY=no
    for (( attempt=1; attempt<=40; attempt++ )); do
        if "${DOCKER[@]}" exec "$CONTAINER" "$PY" -c "$HEALTH_CODE" "$CHECK_IP" "$PORT" >/dev/null 2>&1; then READY=yes; break; fi
        sleep 1
    done
    if [[ "$READY" != yes ]]; then
        echo "API启动未成功，最近日志：" >&2
        "${DOCKER[@]}" exec "$CONTAINER" tail -n 80 "$STATE_DIR/server.log"
        exit 1
    fi
fi
ACCESS_IP="$SERVICE_IP"
[[ "$SERVICE_IP" != 0.0.0.0 ]] || ACCESS_IP=127.0.0.1
if command -v curl >/dev/null; then
    READY=no
    for (( attempt=1; attempt<=10; attempt++ )); do
        if curl --noproxy '*' --fail --silent --max-time 3 "http://$ACCESS_IP:$PORT/healthz" >/dev/null; then READY=yes; break; fi
        sleep 1
    done
    [[ "$READY" == yes ]] || { echo "容器API正常，但宿主端口验证失败，请检查Docker原生映射及防火墙。" >&2; exit 1; }
fi
echo "服务就绪：监听=$SERVICE_IP:$PORT，Docker网络=$MODE；外部访问 http://<服务器实际IP>:$PORT"
echo "Swagger：/docs；给前端AI的完整规范：/api-spec；机器Schema：/openapi.json"
echo "读取Key：sudo docker exec $CONTAINER cat $STATE_DIR/api_key"
echo "API日志：sudo docker exec $CONTAINER tail -f $STATE_DIR/server.log"
echo "训练情况：GET /api/v1/jobs/{id}/metrics、/logs、/events（需Bearer Key）"
echo "持久目录：$STATE_DIR；启动PID记录：$STATE_DIR/server_process.json"
echo "更新代码前先结束活跃任务，再依据PID记录停止API，重新执行此脚本。"
