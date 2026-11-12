#!/usr/bin/env bash
# Linux宿主机运行；与5_verl_api.py放在同一目录，只输入三个位置参数。
# bash 5_verl_api.sh 0.0.0.0 7877 /workspace/models/Qwen3-8B
# 第三个参数也可以是宿主机路径/home/zs/models/Qwen3-8B：优先复用挂载，否则复制。
# 不重建已有训练容器，不安装宿主机Python，不执行uv sync、不升级torch/CUDA。
# 如旧容器没有-p映射，复用其本机镜像创建Python标准库TCP转发容器。
# 转发容器不申请GPU、不加载模型；共享已有镜像层，--pull=never禁止下载镜像。
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
        # shellcheck disable=SC2016 # Docker Go模板，不是shell变量。
        read -r NETWORK TARGET < <("${DOCKER[@]}" inspect -f '{{range $name,$net := .NetworkSettings.Networks}}{{printf "%s %s\n" $name $net.IPAddress}}{{end}}' "$CONTAINER" | head -n 1)
        [[ -n "${NETWORK:-}" && -n "${TARGET:-}" ]] || { echo "无法取得容器bridge地址，请检查网络。" >&2; exit 2; }
        PROXY="verl-api-forward-$PORT"
        # .Image是本机精确镜像ID，不访问镜像仓库，也不复制模型或训练环境。
        PROXY_IMAGE="$("${DOCKER[@]}" inspect -f '{{.Image}}' "$CONTAINER")"
        FORWARD_SPEC="python-v1|$PROXY_IMAGE|$NETWORK|$TARGET|$SERVICE_IP|$PORT"
        REUSE=no
        if "${DOCKER[@]}" inspect "$PROXY" >/dev/null 2>&1; then
            OWNER="$("${DOCKER[@]}" inspect -f '{{index .Config.Labels "com.verl-api.target"}}' "$PROXY")"
            [[ "$OWNER" == "$CONTAINER" ]] || { echo "同名容器不属于本脚本，未修改。" >&2; exit 2; }
            EXISTING_SPEC="$("${DOCKER[@]}" inspect -f '{{index .Config.Labels "com.verl-api.forward-spec"}}' "$PROXY")"
            if [[ "$EXISTING_SPEC" == "$FORWARD_SPEC" ]] && \
               [[ $("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$PROXY") == true ]]; then
                REUSE=yes
                echo "复用现有端口转发容器$PROXY。"
            fi
        fi
        if [[ "$REUSE" != yes ]]; then
            echo "检查已有本机镜像中的Python，仅使用本地镜像，不下载..."
            # 只需标准库，独立检查基础镜像；原容器可写层的.venv不一定在镜像中。
            # shellcheck disable=SC2016 # Python路径探测在新容器的sh中展开。
            if ! PROXY_PY="$("${DOCKER[@]}" run --rm --pull=never --network none \
                -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
                --security-opt no-new-privileges --cap-drop ALL --entrypoint /bin/sh \
                "$PROXY_IMAGE" -c '
for candidate in python3 python /usr/bin/python3 /usr/local/bin/python3 /opt/conda/bin/python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import asyncio,sys; assert sys.version_info >= (3,8)" >/dev/null 2>&1; then
        command -v "$candidate"
        exit 0
    fi
done
exit 1
')"; then
                echo "原镜像中未找到Python 3.8+（容器可写层可能后来安装过Python）。API仍在运行；本脚本没有下载任何转发镜像。" >&2
                exit 1
            fi
            # 透明TCP双向转发：不解析HTTP、不缓冲完整响应，支持SSE和大文件上传。
            # stdin不用于传脚本，避免docker -d导致内容未送达；代码直接作为-c参数。
            PROXY_CODE="$(cat <<'PYPROXY'
import asyncio
import contextlib
import sys

target, port = sys.argv[1], int(sys.argv[2])

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

async def main():
    # 固定容器内监听端口17877，宿主机端口由-p控制；目标仍为API的PORT。
    server = await asyncio.start_server(forward, '0.0.0.0', 17877)
    print('TCP forward ready: :17877 -> %s:%s' % (target, port), flush=True)
    async with server:
        await server.serve_forever()

asyncio.run(main())
PYPROXY
)"
            # 新实现预检成功后才替换旧转发；不操作verl训练容器。
            if "${DOCKER[@]}" inspect "$PROXY" >/dev/null 2>&1; then
                "${DOCKER[@]}" rm -f "$PROXY" >/dev/null
            fi
            echo "创建端口转发容器$PROXY（复用已有镜像，不占GPU）。"
            "${DOCKER[@]}" run -d --pull=never --name "$PROXY" --restart unless-stopped \
                --network "$NETWORK" -p "$SERVICE_IP:$PORT:17877" \
                -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
                --label "com.verl-api.target=$CONTAINER" --label "com.verl-api.forward-spec=$FORWARD_SPEC" \
                --security-opt no-new-privileges --cap-drop ALL \
                --log-opt max-size=10m --log-opt max-file=2 --entrypoint "$PROXY_PY" \
                "$PROXY_IMAGE" -u -c "$PROXY_CODE" "$TARGET" "$PORT" >/dev/null
            # 即使宿主机没有curl，也验证转发进程及其本地端口到API的完整链路。
            READY=no
            for (( attempt=1; attempt<=10; attempt++ )); do
                if "${DOCKER[@]}" exec "$PROXY" "$PROXY_PY" -c "$HEALTH_CODE" 127.0.0.1 17877 >/dev/null 2>&1; then
                    READY=yes
                    break
                fi
                sleep 1
            done
            if [[ "$READY" != yes ]]; then
                echo "转发启动失败；API仍在运行，转发容器日志如下：" >&2
                "${DOCKER[@]}" logs --tail 60 "$PROXY" >&2 || true
                exit 1
            fi
        fi
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
