#!/usr/bin/env bash
# Linux宿主机运行；与5_verl_api.py放在同一目录，只输入三个位置参数。
# bash 5_verl_api.sh 0.0.0.0 7877 /workspace/models/Qwen3-8B
# 第三个参数也可以是宿主机路径/home/zs/models/Qwen3-8B：优先复用挂载，否则复制。
# v5：每次执行都停止现有API及本数据目录的任务，无条件部署同目录py并重启API。
# 训练/推理/数据处理/排队任务会取消；保留数据、历史日志、已有检查点和API Key。
# host网络：第二个参数直接决定监听端口；不重启容器，不创建TCP转发。
# bridge网络：先创建-p IP:PORT:PORT映射；缺少映射则明确报错，不暗中创建转发。
# 新代码先在容器内做语法检查；停止任务不保证生成新的检查点。
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
# 每次都上传，不做新旧版本比较。暂存文件避免覆盖仍被旧worker读取的源码。
STAGED="$REMOTE_DIR/.5_verl_api.deploy.$$.${RANDOM}.py"
cleanup_stage() {
    "${DOCKER[@]}" exec "$CONTAINER" rm -f -- "$STAGED" >/dev/null 2>&1 || true
}
trap cleanup_stage EXIT
"${DOCKER[@]}" cp "$HERE/5_verl_api.py" "$CONTAINER:$STAGED"
echo "准备强制更新：将停止现有API及其训练、推理、数据处理和排队任务。"
"${DOCKER[@]}" exec -i "$CONTAINER" "$PY" - "$STATE_DIR" "$REMOTE_DIR/5_verl_api.py" "$STAGED" "$INNER_IP" "$PORT" "$MODEL" "$VERL_DIR" <<'PYDEPLOY'
import datetime
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request

root, script, staged, bind, port, model, workdir = sys.argv[1:]
root, script, staged = Path(root).resolve(), Path(script).resolve(), Path(staged)
model = str(Path(model).resolve())
done = {'succeeded', 'failed', 'cancelled', 'interrupted'}
database = root / 'state.sqlite3'
record = root / 'server_process.json'

# 串行化同一数据目录的部署；语法或模型路径有问题时不停止正在运行的服务。
deploy_lock = (root / 'deploy.lock').open('a')
try:
    fcntl.flock(deploy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('另一个启动脚本正在更新此API，请等它完成后再运行。')
compile(staged.read_bytes(), str(script), 'exec')

def connect():
    return sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=30)

def jobs():
    if not database.exists():
        return []
    with connect() as db:
        return [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM objects WHERE kind='job'")]

if database.exists():
    with connect() as db:
        row = db.execute("SELECT payload FROM objects WHERE id='base' AND kind='model'").fetchone()
        if row and str(Path(json.loads(row[0])['path']).resolve()) != model:
            sys.exit('此数据目录base已绑定其他模型路径；请用/models/register添加模型。')

def proc(pid):
    try:
        base = Path('/proc') / str(pid)
        raw = (base / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': int(pid), 'start': fields[19], 'ppid': int(fields[1]),
                'group': int(fields[2]), 'session': int(fields[3])}
    except (OSError, ValueError, IndexError):
        return None

def snapshot():
    result = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            p = proc(path.name)
            if p:
                result[p['pid']] = p
    return result

def same(p):
    current = proc(p['pid'])
    return current is not None and current['start'] == p['start']

def send(p, sig):
    # 只处理已核对启动时间的进程，忽略陈旧PID记录。
    if same(p):
        try:
            os.kill(p['pid'], sig)
        except ProcessLookupError:
            pass

initial_jobs = jobs()
job_ids = {j['id'] for j in initial_jobs}
targets, servers, sessions = {}, {}, set()
info = json.loads(record.read_text()) if record.exists() else {}
scripts = {str(script)}
if info.get('script'):
    scripts.add(str(Path(info['script']).resolve()))

def discover():
    table = snapshot()
    for p in table.values():
        pid = p['pid']
        if pid in (1, os.getpid(), os.getppid()):
            continue
        base = Path('/proc') / str(pid)
        try:
            # 只读取身份所需环境字段，不打印环境变量或鉴权信息。
            env = dict(v.split(b'=', 1) for v in (base / 'environ').read_bytes().split(b'\0') if b'=' in v)
            data_root = Path(os.fsdecode(env.get(b'VERL_API_DATA_DIR', b'/workspace/verl_api_data'))).resolve()
            if data_root != root:
                continue
            mode = env.get(b'VERL_API_MODE', b'serve')
            args = [os.fsdecode(a) for a in (base / 'cmdline').read_bytes().split(b'\0') if a]
            # Python程序路径必须是独立argv；不使用pkill -f或模糊字符串匹配。
            cwd = base / 'cwd'
            matches = any(str((Path(a) if Path(a).is_absolute() else cwd / a).resolve()) in scripts
                          for a in args[1:] if a.endswith('.py'))
            owned_job = os.fsdecode(env.get(b'VERL_API_JOB', b'')) in job_ids
            if matches or owned_job:
                targets[pid] = p
                if matches and mode == b'serve':
                    servers[pid] = p
                if mode == b'worker' and p['session'] == pid:
                    sessions.add(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    # 登记PID还需启动时间一致；PID复用不会导致误杀。
    for j in initial_jobs:
        p = table.get(j.get('pid'))
        if p and j.get('pid_start') and p['start'] == str(j['pid_start']):
            targets[p['pid']] = p
            if p['session'] == p['pid']:
                sessions.add(p['pid'])
    # 即使worker先退出，已记录的后代和独立任务session仍会被清理。
    changed = True
    while changed:
        changed = False
        live_ids = {pid for pid, p in targets.items()
                    if pid in table and table[pid]['start'] == p['start']}
        for pid, p in table.items():
            if pid in (1, os.getpid(), os.getppid()):
                continue
            if p['ppid'] in live_ids or p['session'] in sessions:
                if pid not in targets or targets[pid]['start'] != p['start']:
                    targets[pid] = p
                    changed = True

discover()
# 先冻结API，禁止继续接收任务和调度排队任务，再快照其后代并强制结束API。
try:
    for p in servers.values():
        send(p, signal.SIGSTOP)
    discover()
finally:
    for p in servers.values():
        send(p, signal.SIGKILL)

# 补入旧API停止前最后一刻登记的任务，覆盖调度与读取任务表之间的时间差。
known = {j['id'] for j in initial_jobs}
initial_jobs.extend(j for j in jobs() if j['id'] not in known)
job_ids.update(j['id'] for j in initial_jobs)

# worker可以短暂清理资源；不响应TERM的训练rank/vLLM子进程随后强制KILL。
terminated = set()
deadline = time.monotonic() + 3
while True:
    discover()
    live = [p for p in targets.values() if same(p)]
    for p in live:
        identity = (p['pid'], p['start'])
        if identity not in terminated:
            send(p, signal.SIGTERM)
            send(p, signal.SIGCONT)
            terminated.add(identity)
    if not live or time.monotonic() >= deadline:
        break
    time.sleep(.1)
deadline = time.monotonic() + 5
while True:
    discover()
    live = [p for p in targets.values() if same(p)]
    if not live:
        break
    for p in live:
        send(p, signal.SIGKILL)
    if time.monotonic() >= deadline:
        sys.exit('仍有任务进程未退出，未覆盖旧代码或启动新API；请检查容器进程。')
    time.sleep(.1)

# 若有无法识别的API仍持有服务锁，停止部署，不假报重启成功。
server_lock = (root / 'server.lock').open('a')
try:
    fcntl.flock(server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('服务锁仍被占用，未覆盖源码或启动新API。')

# 进程退出后更新持久状态，防止新API重新调度旧队列。
cancelled = 0
was_active = {j['id'] for j in initial_jobs if j['state'] not in done}
if database.exists():
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        for ident, kind, value in db.execute(
                "SELECT id,kind,payload FROM objects WHERE kind IN ('job','chat')").fetchall():
            obj = json.loads(value)
            if obj['state'] in done and not (ident in was_active and obj['state'] == 'failed'):
                continue
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            fields = {'state': 'cancelled', 'finished_at': now,
                      'error': {'code': 'SERVICE_RESTART', 'message': '启动脚本强制更新服务，已取消此任务'}}
            if kind == 'job':
                fields['wait_reason'] = None
            obj.update(fields, updated_at=now)
            db.execute('UPDATE objects SET state=?,payload=? WHERE id=?',
                       ('cancelled', json.dumps(obj, ensure_ascii=False), ident))
            db.execute('INSERT INTO events(ref,kind,ts,payload) VALUES(?,?,?,?)',
                       (ident, 'state', now, json.dumps(fields, ensure_ascii=False)))
            cancelled += 1
print('旧API和任务进程已停止；已取消%d条未完成任务/对话记录。' % cancelled, flush=True)

# 无条件原子替换；不会让启动中的解释器读到只复制了一半的源码。
os.chmod(staged, 0o644)
os.replace(staged, script)
if record.exists():
    record.unlink()
env = dict(os.environ, VERL_API_DATA_DIR=str(root), VERL_API_MODE='serve', PYTHONUNBUFFERED='1')
env.pop('VERL_API_JOB', None)
soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
try:
    resource.setrlimit(resource.RLIMIT_STACK, (64 * 1024 * 1024, hard))
except (OSError, ValueError):
    pass
fcntl.flock(server_lock, fcntl.LOCK_UN)
server_lock.close()
with (root / 'server.log').open('a') as log:
    log.write('\n=== 强制部署并启动API %s ===\n' % datetime.datetime.now(datetime.timezone.utc).isoformat())
    log.flush()
    child = subprocess.Popen([sys.executable, '-u', str(script), bind, port, model],
                             cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
check_ip = '127.0.0.1' if bind == '0.0.0.0' else bind
for _ in range(40):
    if child.poll() is not None:
        break
    try:
        with opener.open('http://%s:%s/healthz' % (check_ip, port), timeout=1) as response:
            health = json.load(response)
        started = json.loads(record.read_text())
        p = proc(child.pid)
        if (health.get('service') == 'verl-api' and started.get('pid') == child.pid
                and p and str(started.get('pid_start')) == p['start']):
            print('已使用本次上传的5_verl_api.py启动新API，PID=%d。' % child.pid, flush=True)
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    sys.exit('新API健康检查超时，请查看server.log。')
if child.poll() is not None:
    sys.exit('新API已退出（exit=%s），请查看server.log。' % child.returncode)
PYDEPLOY
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
echo "每次运行本脚本都会取消现有任务、覆盖同目录py并重启API；已有数据和检查点保留。"
