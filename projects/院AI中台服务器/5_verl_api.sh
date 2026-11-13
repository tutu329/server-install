#!/usr/bin/env bash
# Linux宿主机运行；与5_verl_api.py放在同一目录，只输入三个位置参数。
# bash 5_verl_api.sh 0.0.0.0 7877 /workspace/models/Qwen3-8B
# 第三个参数也可以是宿主机路径/home/zs/models/Qwen3-8B：优先复用挂载，否则复制。
# 不重建已有训练容器，不安装宿主机Python，不执行uv sync、不升级torch/CUDA。
# v3：旧容器没有-p映射时，使用sudo nsenter复用运行中容器的Python做宿主机TCP转发。
# 不新建/下载镜像，不安装宿主机Python，不加载模型；需要宿主机util-linux的nsenter。
# 脚本后台启动API，重复运行保留现有API/任务，仅检查或修复转发。
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
MODE="$("${DOCKER[@]}" inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER")"
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
    echo "API已经运行，保留现有代码和任务；检查端口接入。"
else
    # 活跃服务运行时不覆盖源文件，避免正在训练的rank导入不同版本。
    "${DOCKER[@]}" cp "$HERE/5_verl_api.py" "$CONTAINER:$REMOTE_DIR/5_verl_api.py"
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
if [[ "$MODE" != host ]]; then
    PUBLISHED="$("${DOCKER[@]}" port "$CONTAINER" "$PORT/tcp" 2>/dev/null || true)"
    if [[ -n "$PUBLISHED" ]]; then
        MAPPING_OK=no
        while IFS= read -r binding; do
            if [[ "$binding" == "$SERVICE_IP:$PORT" || "$binding" == "0.0.0.0:$PORT" ]]; then
                MAPPING_OK=yes
            fi
        done <<< "$PUBLISHED"
        if [[ "$MAPPING_OK" != yes ]]; then
            echo "已有不同端口映射: $PUBLISHED；请使用匹配的IP/端口或调整原映射。" >&2; exit 2
        fi
        echo "复用原端口映射: $PUBLISHED"
    else
        # Linux宿主机保留自身network namespace，仅进入运行中容器的mount/PID空间。
        # 因而可使用容器可写层及挂载中的Python，在宿主机监听端口；没有新镜像。
        NSENTER_PATH="$(command -v nsenter || true)"
        [[ -n "$NSENTER_PATH" ]] || { echo "宿主机缺少nsenter（util-linux）；API仍在运行。" >&2; exit 2; }
        ROOT_CMD=()
        if (( EUID != 0 )); then
            command -v sudo >/dev/null || { echo "端口转发需要sudo运行nsenter。" >&2; exit 2; }
            ROOT_CMD=(sudo)
            "${ROOT_CMD[@]}" -v
        fi
        CONTAINER_PID="$("${DOCKER[@]}" inspect -f '{{.State.Pid}}' "$CONTAINER")"
        [[ "$CONTAINER_PID" =~ ^[1-9][0-9]*$ ]] || { echo "未取得运行中容器PID。" >&2; exit 2; }
        HOST_NET="$(readlink /proc/self/ns/net)"
        ENTER=("${ROOT_CMD[@]}" "$NSENTER_PATH" --target "$CONTAINER_PID" --mount --pid --root --wd --)
        # 这里有意不加--net，否则监听端口仍落在容器里。
        echo "检查运行中容器的Python和宿主机网络（不使用原始镜像）..."
        if ! "${ENTER[@]}" "$PY" -I -c 'import asyncio,fcntl,os,sys; assert os.readlink("/proc/self/ns/net")==sys.argv[1], "未保留宿主机网络"; print("已使用运行中容器Python:",sys.executable)' "$HOST_NET"; then
            echo "nsenter检查失败，以上为原始错误；需要本机Linux Docker及sudo权限。API仍在运行。" >&2
            exit 1
        fi
        # shellcheck disable=SC2016 # Docker Go模板，不是shell变量。
        TARGET="$("${DOCKER[@]}" inspect -f '{{range .NetworkSettings.Networks}}{{if .IPAddress}}{{println .IPAddress}}{{end}}{{end}}' "$CONTAINER" | head -n 1)"
        [[ -n "$TARGET" ]] || { echo "容器没有可供宿主机连接的IPv4地址。" >&2; exit 2; }
        FORWARD_FILE="$STATE_DIR/port_forward.py"
        # 写入独立运行时辅助文件；主API的5_verl_api.py不变。
        "${DOCKER[@]}" exec -i "$CONTAINER" "$PY" -I -c 'import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); q=p.with_suffix(".tmp"); q.write_bytes(sys.stdin.buffer.read()); q.chmod(0o600); os.replace(q,p)' "$FORWARD_FILE" <<'PYFORWARD'
import asyncio
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
import urllib.request

mode, bind, port_text, target, root_text, host_net = sys.argv[1:]
port = int(port_text)
root = Path(root_text)
record = root / ('port_forward_%s.json' % port)
lock_path = root / ('port_forward_%s.lock' % port)
log_path = root / ('port_forward_%s.log' % port)
spec = {'version': 3, 'bind': bind, 'port': port, 'target': target, 'host_net': host_net}

def start_tick(pid):
    try:
        text = Path('/proc/%s/stat' % pid).read_text()
        fields = text[text.rfind(')') + 2:].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None

def load_record():
    try:
        return json.loads(record.read_text())
    except (OSError, ValueError):
        return {}

def live(info):
    return bool(info.get('pid_start') and start_tick(info.get('pid')) == info['pid_start'])

def health(host):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://%s:%s/healthz' % (host, port), timeout=4) as response:
        if json.load(response).get('service') != 'verl-api':
            raise RuntimeError('目标端口不是verl-api')

async def copy(reader, writer):
    while True:
        data = await reader.read(65536)
        if not data:
            if writer.can_write_eof():
                writer.write_eof()
                await writer.drain()
            return
        writer.write(data)
        await writer.drain()

async def forward(reader, writer):
    upstream = None
    tasks = []
    try:
        remote, upstream = await asyncio.wait_for(asyncio.open_connection(target, port), 10)
        tasks = [asyncio.create_task(copy(reader, upstream)), asyncio.create_task(copy(remote, writer))]
        await asyncio.gather(*tasks)
    except (OSError, asyncio.TimeoutError) as exc:
        print('Forward connection ended: ' + str(exc), flush=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for stream in (writer, upstream):
            if stream is not None:
                stream.close()
                with contextlib.suppress(Exception):
                    await stream.wait_closed()

async def serve(listener):
    server = await asyncio.start_server(forward, sock=listener)
    print('TCP forward ready: %s:%s -> %s:%s' % (bind, port, target, port), flush=True)
    async with server:
        await server.serve_forever()

def main():
    if mode == 'status':
        info = load_record()
        if not live(info) or info.get('spec') != spec:
            raise RuntimeError('转发进程尚未就绪或配置不匹配')
        if os.readlink('/proc/self/ns/net') != host_net:
            raise RuntimeError('健康检查未在宿主机网络执行')
        health('127.0.0.1' if bind == '0.0.0.0' else bind)
        print('宿主机端口转发健康检查通过。')
        return
    if mode != 'ensure':
        raise RuntimeError('不支持的内部操作')
    if os.readlink('/proc/self/ns/net') != host_net:
        raise RuntimeError('当前进程不在宿主机网络，拒绝启动')
    health(target)
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        info = load_record()
        if live(info) and info.get('spec') == spec:
            print('复用正在运行的宿主机端口转发。')
            return
        # 只停止持有此服务状态记录且创建时间匹配的旧转发进程。
        if not live(info):
            raise RuntimeError('转发锁已占用但状态不可确认，请稍后重试')
        os.kill(info['pid'], signal.SIGTERM)
        for _ in range(50):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        else:
            raise RuntimeError('旧转发尚未退出，未强制终止')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind, port))  # 先绑定再后台化：端口冲突直接显示给调用者。
    listener.listen(128)
    listener.setblocking(False)
    if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
        log_path.replace(log_path.with_suffix('.log.1'))
    log = log_path.open('a', buffering=1)
    child = os.fork()
    if child:
        print('已启动宿主机端口转发；容器内PID: %s' % child)
        return
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with open(os.devnull, 'rb') as null:
        os.dup2(null.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    info = {'pid': os.getpid(), 'pid_start': start_tick(os.getpid()), 'spec': spec}
    temp = record.with_suffix('.tmp')
    temp.write_text(json.dumps(info))
    temp.chmod(0o600)
    temp.replace(record)
    try:
        asyncio.run(serve(listener))
    finally:
        # lock对象保持存活，进程退出时内核自动释放文件锁。
        lock.close()

try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
PYFORWARD
        # 清理前两版脚本创建的转发容器，仅限本脚本带所有权标签的对象。
        LEGACY_PROXY="verl-api-forward-$PORT"
        if "${DOCKER[@]}" inspect "$LEGACY_PROXY" >/dev/null 2>&1; then
            OWNER="$("${DOCKER[@]}" inspect -f '{{index .Config.Labels "com.verl-api.target"}}' "$LEGACY_PROXY")"
            [[ "$OWNER" == "$CONTAINER" ]] || { echo "同名转发容器不属于本脚本，未修改。" >&2; exit 2; }
            "${DOCKER[@]}" rm -f "$LEGACY_PROXY" >/dev/null
        fi
        echo "启动宿主机$SERVICE_IP:$PORT转发，复用运行中容器的Python..."
        "${ENTER[@]}" "$PY" -I -u "$FORWARD_FILE" ensure "$SERVICE_IP" "$PORT" "$TARGET" "$STATE_DIR" "$HOST_NET"
        READY=no
        for (( attempt=1; attempt<=10; attempt++ )); do
            if "${ENTER[@]}" "$PY" -I "$FORWARD_FILE" status "$SERVICE_IP" "$PORT" "$TARGET" "$STATE_DIR" "$HOST_NET" >/dev/null 2>&1; then
                READY=yes
                break
            fi
            sleep 1
        done
        if [[ "$READY" != yes ]]; then
            "${DOCKER[@]}" exec "$CONTAINER" tail -n 60 "$STATE_DIR/port_forward_$PORT.log" >&2 || true
            echo "宿主机转发检查失败，原API仍在运行。" >&2
            exit 1
        fi
        echo "转发日志：sudo docker exec $CONTAINER tail -f $STATE_DIR/port_forward_$PORT.log"
        echo "容器重启后，重新执行本脚本以恢复API和转发。"
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
    [[ "$READY" == yes ]] || { echo "容器API正常，但宿主端口验证失败，请检查转发容器及防火墙。" >&2; exit 1; }
fi
echo "服务就绪：外部访问 http://<服务器实际IP>:$PORT"
echo "Swagger：/docs；给前端AI的完整规范：/api-spec；机器Schema：/openapi.json"
echo "读取Key：sudo docker exec $CONTAINER cat $STATE_DIR/api_key"
echo "API日志：sudo docker exec $CONTAINER tail -f $STATE_DIR/server.log"
echo "训练情况：GET /api/v1/jobs/{id}/metrics、/logs、/events（需Bearer Key）"
echo "持久目录：$STATE_DIR；启动PID记录：$STATE_DIR/server_process.json"
echo "更新代码前先结束活跃任务，再依据PID记录停止API，重新执行此脚本。"
