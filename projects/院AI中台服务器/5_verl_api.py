#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verl Workbench — 容器内单文件CPT/SFT、GPU监控和模型对话。

启动：python 5_verl_api.py [IP=0.0.0.0] [PORT=7877] [MODEL=/workspace/models/Qwen3-8B]
仅三个位置参数。宿主机用同目录5_verl_api.sh启动。完整接口规范在下方SPEC和请求模型中，
运行后GET /api-spec、/openapi.json、/docs可供另一个AI编制跨节点前端。
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import io
import ipaddress
import itertools
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = "1.1.1"
SCRIPT = Path(__file__).resolve()
ROOT = Path(os.environ.get("VERL_API_DATA_DIR", "/workspace/verl_api_data")).resolve()
MODEL_DEFAULT = "/workspace/models/Qwen3-8B"
LIVE = {"starting", "running", "loading", "ready", "stopping"}
DONE = {"succeeded", "failed", "cancelled", "interrupted"}
SETTINGS = {
    "cors_origins": ["*"],
    "gpu_poll_seconds": 1.0,
    "auto_unload_inference": True,
    "max_cpu_jobs": 1,
    "max_upload_bytes": 5 * 1024**3,
}
SPEC = r"""
# 5_verl_api.py 完整接口规范 v1.1

## 部署与范围
在已有verl环境的容器中运行，只需IP、PORT、MODEL三个位置参数。
默认0.0.0.0 7877 /workspace/models/Qwen3-8B。浏览器使用真实服务器IP，而非0.0.0.0。
数据目录/workspace/verl_api_data，需由/workspace挂载实现持久化。
代码是一个文件；运行生成数据库、数据集、日志、检查点及模型是正常行为。
单机NVIDIA GPU，FSDP/FSDP2；CPT/SFT均支持full/LoRA；文本模型，以Qwen3-8B为适配基线。
API单worker，重型任务在独立进程；API重启不会主动杀掉训练，按PID+进程创建时间接管。
本版不提供RL/OPD、多节点训练、视觉输入、tool calling或任意shell执行接口。

## 磁盘查看与手动清理（v1.1）
所有 /api/v1/storage/* 接口沿用 Bearer Key。仅管理容器可见文件；不扫描宿主机其他目录。
GET /storage/summary：volumes为按设备去重的磁盘total/used/free_bytes，categories为已发现产物分类占用。
磁盘已用包含其他程序，不等于本API占用。分类统计不含数据库/环境缓存等，跨产物硬链接可能重复计数。
首次state=not_scanned。POST /storage/scan返回202，后台扫描；轮询summary直到succeeded/failed。
扫描结果包含scanned_at、error；无需每次页面刷新重新扫描。扫描不跟随符号链接、不读取权重内容。
GET /storage/artifacts?type=checkpoint&job_id=...&min_bytes=0&sort=size_desc&offset=0&limit=50
返回items/total/offset/limit/scanned_at。type可为checkpoint/model/residual/dataset_version/snapshot/log/raw_dataset。
每项有artifact_id/type/path/job_id/object_ids/step、size_bytes、allocated_bytes、file_count、modified_at、
deletable/blocked_reasons/active_job_ids、scan_errors；删除权限为扫描时快照，不保证之后可删除。
GET /storage/artifacts/{artifact_id}/files?offset=0&limit=100：实时扫描单个产物，按大小降序返回相对路径、大小。
预览 POST /storage/cleanup/preview：
1. {"mode":"selected","artifact_ids":["artifact_..."]} 精确选择完整产物；不接受任意路径。
2. {"mode":"old_checkpoints","job_id":"job_...","keep_last":2} 保留step最高的2个检查点。
3. {"mode":"job_outputs","job_id":"job_..."} 删除该任务的检查点/模型/失败输出；日志和loss不删。
导出模型属于export任务；清理原训练任务不会顺带清理其独立export任务的模型。
返回计划id、items、executable、estimated_bytes、effects和expires_at(Unix秒)，有效15分钟。
存在任何阻止项时executable=false，不会偷偷跳过后执行其余项；请重新选择。
执行 POST /storage/cleanup {"plan_id":"storage_plan_...","confirm":true} 返回202和清理记录id。
轮询 GET /storage/cleanup/{id}，state=queued/running/succeeded/failed/interrupted；
items逐项报告成功/失败/error，completed/total是产物数，deleted_bytes是完整删除项分配大小估计。
free_bytes_delta是数据盘可用空间实测差值，包含并发写入影响，不保证等于deleted_bytes。
计划一次性使用；重复提交、过期、文件变化或新增依赖返回409，必须重新预览。
真正unlink/rmtree释放文件，不设回收站；不删除任务历史、数据库、API Key、原始数据、日志和loss。
保护base、外部注册模型、符号链接目录以及活跃/排队任务引用的产物；不自动停止任务。
现存LoRA模型的底座和数据版本预览依赖的tokenizer也受保护；历史来源checkpoint关系不阻止独立模型保留。
清理期间数据/模型文件读取与对象操作、任务创建/验证、adapter加载返回STORAGE_BUSY(409)，已有无关任务继续执行。
删除检查点state=missing,resumable=false；模型/数据版本state=deleted；storage_state记录删除状态。
被删除/不完整产物再次用于任务返回ARTIFACT_UNAVAILABLE(409)，记录保留用于追溯。
服务重启不会自动重放删除；中断清理state=interrupted，重新扫描、预览后才能继续。
前端顺序：summary→必要时scan→artifacts→files→preview展示影响→用户确认→cleanup→轮询→刷新列表。

### 删除数据对象与历史记录
GET /storage/records?type=dataset&offset=0&limit=50 提供可选择的对象记录。
type=dataset/dataset_version/train_job/checkpoint/model/inference/other_job；记录不存在文件时仍可清理。
POST /storage/cleanup/preview {"mode":"records","record_ids":["dataset_...","dv_...","job_...","ckpt_...","model_..."]}
返回records（完整最终删除列表）、blocked_records、items（文件清单）、executable、effects。
执行仍使用 POST /storage/cleanup {"plan_id":"...","confirm":true}，轮询同一状态接口。
删除数据集需先删除或同时选中其全部数据版本；不会静默级联其他独立任务。
删除任务时，所属checkpoint、导出模型或准备的数据版本自动明确列入预览，依赖检查仍适用。
删除训练任务不会删除由独立export任务产生的模型；删除推理历史只允许已结束且进程已退出的实例。
重要区别：普通selected/job_outputs/old_checkpoints文件清理保留日志和loss；
records模式删除任务/推理历史会永久删除该任务记录、日志、loss、事件及任务自有目录。
数据集原始文件仅在records模式明确删除数据集时才可删除。
base和外部注册模型继续受保护；已被其他模型/检查点依赖的底座不能删除。
删除记录成功后，原列表不再返回该记录、按ID访问返回404；原始任务的历史来源ID可能保留为追溯信息。
文件部分删除失败时保留记录，标记不完整；不会把未成功删除的对象从数据库消失。重新预览可重试剩余部分。
清理审核计划和操作结果保留；SQLite页可能复用，删除历史不保证数据库文件立即缩小。

### 进程占用核验（v1.1.1）
调度、卸载和删除使用统一job_process_status：status=live/unknown/exited。
僵尸Z和已退出X进程不算占用；进程组存在本身不证明任务存活。
通过PID+启动时间、已验证session或VERL_API_JOB标记确认归属，不输出环境变量内容。
预检blocked_records/process_blockers附PID、状态、PGID/session和归属依据；unknown仍保护文件，但明确说明无法核验。
重复卸载终态实例会重新核验；有已证实残留则进入stopping并以pidfd停止，不能误杀复用PID/未知进程。
卸载返回200表示请求已处理，不等于所有进程已退出；检查state及process_status，停止后重新预检。

## 认证/网络/通用结构
首次启动自动生成/workspace/verl_api_data/api_key（文件0600）。无需第四启动参数。
除/healthz、/api-spec、/openapi.json、/docs、/redoc外均需Authorization: Bearer <KEY>。
Swagger使用Authorize。Key不要放URL；推荐z-desktop后端代理保管Key。
CORS默认允许跨源Bearer请求，不使用cookie；PATCH /api/v1/settings可限制cors_origins。
HTTPS前端需要HTTPS反向代理接入。SSE代理关闭缓冲，读取超时大于15秒。
所有路径都是容器路径；所有时间UTC ISO8601，时长秒，内存byte，功耗瓦，温度摄氏度。
错误统一{error:{code,message,details}}；400配置、401认证、404不存在、409状态冲突、
413上传大小、422参数/数据、503环境/GPU/推理不可用。缺失指标为null，不伪造0。
列表{items,total,offset,limit}；一般成功200，异步任务202返回job对象。
job结构：{id,kind:"job",job_type,state,request,result,error,progress,created_at,updated_at,
 gpu_uuids,pid?,started_at?,finished_at?,wait_reason?,log_path,output_dir}。
state：queued -> starting -> running -> succeeded/failed；取消->stopping->cancelled；
失联->interrupted。推理为queued->loading->ready，卸载后cancelled。
对象id不可从名称推测。异步操作轮询GET /api/v1/jobs/{id}或订阅events直到结束。

## 数据准备
POST /api/v1/datasets {name,task_type:"sft"|"cpt"}创建可编辑原始数据集。
POST /api/v1/datasets/{id}/samples {samples:[...]}追加，最多10000条/请求。
GET同路径查看；PUT /samples/{index} {sample:{...}}替换；DELETE /samples/{index}删除。
POST /api/v1/datasets/{id}/upload：multipart字段file，JSON/JSONL/Parquet，CPT另支持TXT/MD。
POST /api/v1/datasets/{id}/import {path:"容器内文件"}先复制快照，返回import job。
JSON应为数组或{samples:[...]}；JSONL每行一条对象。TXT/MD整个文件是一篇CPT文档。
CPT记录：{text:"正文",source_id:"文档ID",metadata:{...}}。
SFT记录：{messages:[{role:"user",content:"问题"},{role:"assistant",content:"答案"}],source_id:"案例ID"}。
兼容instruction/input/output。仅允许system?+user/assistant交替，以assistant结束。
assistant可提供reasoning_content，或content中的完整<think>...</think>；不能同时提供两者。
enable_thinking按样本指定，默认根据是否有非空思考块推断；true必须有思考示范。
metadata不进入正文；本服务不自动生成问答，Word/PDF解析和教师生成可由z-desktop完成。
POST /api/v1/datasets/demo {task_type:"sft"}创建16条不同演示样本。
POST /api/v1/datasets/{id}/prepare {model_id:"base",max_length:1024,validation_ratio:0.05,
 seed:42,deduplicate:true,assistant_scope:"all",overlength:"error",on_invalid:"error"}返回prepare job。
成功result={dataset_version_id,report}。版本不可变，训练绑定版本ID。
先去重并按source_id分组划分train/val，再切块，避免同源文档泄漏；缺失source_id用内容hash。
小样本演示设置validation_ratio=0。按分组划分的实际比例可能与设定值不同，报告如实返回。
CPT文末EOS，长文按token切块、保留一个上下文token衔接，不跨文档混拼；首token不监督。
SFT all将每个assistant回合展开为完整历史+当前答案，每条仅当前答案受监督；last只保留末轮。
完整chat_template与generation prompt严格检查token前缀，不忽略Qwen3模板不一致。
overlength=error/drop/truncate；truncate可能裁掉答案尾部，监督为空则报错或按on_invalid跳过。
on_invalid=error/skip；报告统计原始数、重复数、无效数、切分数、token分布，保留前100条错误。
GET /api/v1/dataset-versions及/{id}查询；/{id}/preview?split=train&offset=0&limit=5返回
input_ids/loss_mask/decoded_text/supervised_text。mask与输入对齐，未提前shift，shift由verl做。
/{id}/files/{name}可下载train.parquet、val.parquet、manifest.json、report.json。

## 训练与参数
POST /api/v1/jobs/validate与POST /api/v1/jobs使用同一TrainRequest。
示例：{task_type:"sft",finetune:"lora",dataset_version_id:"dv_...",model_id:"base",gpu_ids:[0,1,2,3],
 train_batch_size:4,micro_batch_size_per_gpu:1,epochs:1,max_steps:2,
 lora:{rank:16,alpha:32,target_modules:"all-linear"}}。
全参默认lr=1e-5，LoRA=1e-4；rank16/alpha32；FP32主参数+引擎默认BF16混合计算；
FSDP2，梯度检查点开启，torch compile关闭，no_padding，静态batch，数据worker=0。
GPU索引/UUID取自/gpus；内部CUDA_VISIBLE_DEVICES使用UUID。GPU数必须兼容并行参数。
global batch需能被DP大小整除；静态micro batch需整除每DP batch；数据需够一个完整batch。
max_steps受epochs限制，不会自动无限重复；validate返回实际步数上限和警告。
高级verl_overrides使用扁平原生字段，如{"engine.offload_policy":true,"optim.weight_decay":0.01}。
优先级：实际verl默认 < 服务常用字段 < verl_overrides；validate返回resolved_config及启动argv。
模型路径、数据路径/长度/custom_cls、GPU数量、输出、logger、resume由服务管理，不能高级覆盖。
未知字段或未实现的后端拒绝；禁止Hydra动态类、插值和自定义代码。
FSDP当前LoRA的dropout/rsLoRA不开放，避免接受但不生效；支持rank/alpha/target/exclude模块。
runtime_env仅限OMP_NUM_THREADS、NCCL_DEBUG、NCCL_P2P_DISABLE、NCCL_IB_DISABLE、
NCCL_SOCKET_IFNAME、NCCL_SHM_DISABLE、NCCL_NET_GDR_LEVEL、NCCL_P2P_LEVEL。
checkpoint保存model/optimizer/extra；save_freq=-1仍保存最后一步；以完整回调确认后登记。
GET /api/v1/jobs?kind=train&state=running，GET /jobs/{id}查询状态。
POST /jobs/{id}/cancel停止整个已登记进程组，20秒后强制终止；不保证临停时新建checkpoint。
POST /jobs/{id}/resume {checkpoint_id,additional_epochs:1}创建新任务，从完整检查点恢复。
恢复使用原数据和GPU数量；max_steps清空，总epochs增加；不是任意step的无损暂停。

## 训练情况、曲线和日志（前端重要）
GET /api/v1/jobs/{id}/metrics?after=0&limit=1000
返回{items:[{seq,timestamp,step,elapsed_seconds,metrics:{原始verl键值},summary:{loss,validation_loss,
 learning_rate,grad_norm,epoch,tokens_per_second,step_seconds}}],next_after}。
after是事件序号，不是step；step为优化器更新步。指标从verl Tracking.log回调提取，非解析进度条。
历史metrics独立持久化；日志事件滚动清理不影响loss全历史。step_seconds是墙钟间隔，可能包括验证/保存。
GET /jobs/{id}/logs?offset=0&limit=65536 返回{text,next_offset,size,eof}。
offset单位byte，最大读取1MiB/次，UTF-8边界安全；日志保存完整stdout/stderr。
GET /jobs/{id}/events?after=0 为SSE；支持Last-Event-ID。事件state/metric/log/checkpoint/heartbeat/reset。
每帧id: seq、event:类型、data:{seq,ref_id,kind,timestamp,payload}。15秒心跳。
最近50000事件可续传；游标过期收到reset，此时重新GET metrics及logs补历史。
GET /api/v1/events同格式用于所有任务。浏览器用fetch流携带Bearer，EventSource不能设置此header。
GET /api/v1/gpus返回{available,timestamp,gpus,error?}；GPU项包含index/uuid/name、显存byte、
利用率%、温度、功耗、时钟、驱动、进程及reserved_by。不可读值null；GPU利用率不同于MFU。
GET /api/v1/gpus/events按gpu_poll_seconds推送快照，不保存GPU历史。

## 模型、导出、推理实例
GET /api/v1/models为全部登记版本。POST /models/register {name,path,model_kind:"full"|"lora",base_model_id?}。
base是启动模型；同一数据目录重启应保持该路径，其他模型通过register添加。
GET /checkpoints?job_id=...列完整检查点；被保留策略清理的项state=missing。
POST /exports {checkpoint_id,name,format:"auto"|"adapter"|"merged"}返回export job。
全参auto导出HF；LoRA auto/adapter导出PEFT adapter并绑定底座，merged导出合并HF。
导出是CPU任务，需要足够RAM/磁盘；成功result={model_id,path,model_kind}。
GET /models/{id}/files列文件；/files/{relative_path}下载目录内文件。没有任意删除模型接口。
POST /inference {model_id:"base",gpu_ids:[0],tensor_parallel_size:1,dtype:"bfloat16",
 gpu_memory_utilization:0.7,max_model_len:8192,max_num_seqs:2,max_lora_rank:64,enforce_eager:true}
返回inference job，等待ready；GET /inference列表，DELETE /inference/{id}卸载。
默认同一卡训练/推理互斥；训练等待当前生成完成后卸载冲突的本系统实例，外部进程不终止。
指定不同GPU可并行。vLLM仅绑定容器127.0.0.1随机内部端口，统一由7877代理并隐藏内部key。
LoRA实例可选择adapter或同一底座对话；不同全参版本可不同GPU加载或顺序卸载对比。
POST /inference/{id}/adapters {model_id}动态加载同底座adapter；DELETE /adapters/{model_id}卸载。
GET /v1/models仅列ready的对话模型；PATCH /api/v1/models/{id}/chat-defaults设置版本默认参数。

## 对话
POST /v1/chat/completions {model:"base",messages:[{role:"user",content:"你好"}],stream:true,
 enable_thinking:false,temperature:0.7,max_tokens:2048}。
多轮消息为system/user/assistant文本；历史只需最终回答，不需回传先前思考。
请求显式值 > 版本默认 > 模式预设。默认thinking=true；开启时temperature=.6/top_p=.95/top_k=20/
min_p=0/max_tokens=4096；关闭时temperature=.7/top_p=.8/top_k=20/min_p=0/max_tokens=2048。
其余默认repetition_penalty=1、presence_penalty=0、frequency_penalty=0、seed=null、stop=null。
max_tokens包含思考+答案；finish_reason=length可能尚无最终回答。thinking_token_budget只在后端实际支持时允许。
返回reasoning_content和content分别显示思考/答案；兼容vLLM旧reasoning_content及新reasoning字段。
非流式返回OpenAI风格choices/usage，增加request_id、effective_parameters、timing。
流式data:JSON\n\n，末尾data:[DONE]；增量在choices[].delta；额外metadata帧choices为空，
携带effective_parameters或timing，前端必须容忍空choices。错误帧{error:{code,message}}。
响应header X-Request-ID；GET /api/v1/chat/requests/{id}查询，POST同路径/cancel取消当前生成。
聊天流不支持重放；断连会关闭上游请求。系统只保存调用元信息，不持久保存完整聊天正文。
timing含elapsed_seconds、time_to_first_token_seconds、output_tokens_per_second；缺usage时速度null。

## 推荐前端流程
system/capabilities/gpus -> 创建/上传数据 -> prepare -> 预览 -> jobs/validate -> jobs ->
订阅events画loss并用logs显示日志 -> checkpoints -> exports -> inference -> 对话比较。
验收可先datasets/demo，再prepare(max_length=128,validation_ratio=0)，max_steps=2训练。
不要把job返回202当作训练成功；等待state。对比时固定问题、版本和采样参数。
GET /api/v1/config/defaults提供实际verl完整配置。/openapi.json给出每个请求字段类型/范围/默认值。
以下代码中的请求模型和路由即规范的一部分；/api-spec会附加所有请求JSON Schema及路由表。
"""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def ident(prefix):
    return prefix + "_" + uuid.uuid4().hex[:16]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item"):
        try:
            return clean(x.item())
        except (ValueError, RuntimeError):
            pass
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def js(x):
    return json.dumps(
        clean(x), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(js(value))
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


class Error(Exception):
    def __init__(self, message, code="INVALID_CONFIG", status=400, details=None):
        super().__init__(message)
        self.message, self.code, self.status, self.details = (
            message,
            code,
            status,
            details,
        )


def check(ok, message, code="INVALID_CONFIG", status=400):
    if not ok:
        raise Error(message, code, status)


def command(argv, timeout=60, env=None, cwd=None):
    p = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd
    )
    check(
        p.returncode == 0,
        p.stderr[-6000:] or p.stdout[-6000:],
        "SUBPROCESS_FAILED",
        503,
    )
    return p.stdout


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def proc_start(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        f = text[text.rfind(")") + 2 :].split()
        return f[19] if f[0] != "Z" else None
    except (OSError, IndexError):
        return None


def alive(job):
    return bool(job.get("pid_start") and proc_start(job.get("pid")) == job["pid_start"])


def process_snapshot(pid):
    text = Path(f"/proc/{int(pid)}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), session_id=int(fields[3]), start_time=fields[19])


def process_boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def process_job_marker(pid, job_id):
    # Compare only the task marker. Never expose environment contents (keys/secrets).
    values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return ("VERL_API_JOB=" + job_id).encode() in values


def job_process_status(job):
    """Prove identity separately from existence; zombies never hold task resources."""
    live, uncertain, ignored = [], [], []
    pid = job.get("pid")
    started = job.get("pid_start")
    boot = process_boot_id()
    old_boot = job.get("process_boot_id")
    if old_boot and boot and old_boot != boot:
        return dict(status="exited", live_processes=[], uncertain_processes=[],
                    ignored_processes=[{"reason": "previous_boot"}])
    leader = None
    leader_unknown = False
    if pid and started:
        try:
            leader = process_snapshot(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, IndexError):
            leader_unknown = True
    leader_matches = bool(leader and leader["start_time"] == str(started))
    reused = bool(leader and not leader_matches)
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        entries = []
        uncertain.append({"reason": "proc_inventory_unreadable"})
    for entry in entries:
        if not entry.name.isdigit():
            continue
        current_pid = int(entry.name)
        try:
            info = process_snapshot(current_pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            if current_pid in (pid, job.get("child_pid")):
                uncertain.append({"pid": current_pid, "reason": "process_stat_unreadable"})
            continue
        in_group = bool(pid and info["pgid"] == pid and info["session_id"] == pid)
        exact_parent = current_pid == pid and leader_matches
        exact_child = (current_pid == job.get("child_pid") and job.get("child_pid_start")
                       and info["start_time"] == str(job["child_pid_start"]))
        candidate = in_group or exact_parent or exact_child
        if info["state"] in ("Z", "X", "x"):
            if candidate:
                ignored.append({**info, "reason": "zombie_or_dead"})
            continue
        marker, unreadable = False, False
        if job.get("id"):
            try:
                marker = process_job_marker(current_pid, job["id"])
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                unreadable = True
        if marker:
            evidence = "job_environment_marker"
        elif exact_parent:
            evidence = "pid_and_start_time"
        elif exact_child:
            evidence = "child_pid_and_start_time"
        elif in_group and leader_matches and int(info["start_time"]) >= int(started):
            evidence = "verified_worker_session"
        elif candidate and reused:
            ignored.append({**info, "reason": "worker_pid_reused"})
            continue
        elif candidate:
            uncertain.append({**info, "reason": "identity_unverified" if not unreadable else "identity_unreadable"})
            continue
        else:
            continue
        # Recheck identity after reading environ, which can race process exit/reuse.
        try:
            again = process_snapshot(current_pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            uncertain.append({**info, "reason": "identity_recheck_failed"})
            continue
        if again["start_time"] != info["start_time"] or again["state"] in ("Z", "X", "x"):
            continue
        live.append({**again, "evidence": evidence})
    if leader_unknown and not any(x.get("pid") == pid for x in uncertain):
        uncertain.append({"pid": pid, "reason": "worker_identity_unreadable"})
    return dict(status="live" if live else "unknown" if uncertain else "exited",
                live_processes=live, uncertain_processes=uncertain, ignored_processes=ignored)


def stop_process(job, sig=signal.SIGTERM):
    report = job_process_status(job)
    actions = []
    for info in report["live_processes"]:
        fd = None
        try:
            # pidfd pins the process; never signal an unverified/reused whole group.
            check(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                  "当前平台不支持安全进程终止(pidfd)", "PROCESS_IDENTITY_UNVERIFIED", 409)
            fd = os.pidfd_open(info["pid"])
            current = process_snapshot(info["pid"])
            if current["start_time"] != info["start_time"] or current["state"] in ("Z", "X", "x"):
                continue
            signal.pidfd_send_signal(fd, sig)
            actions.append({"pid": info["pid"], "signal": int(sig), "sent": True})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, Error) as e:
            actions.append({"pid": info["pid"], "sent": False, "error": str(e)})
        finally:
            if fd is not None:
                os.close(fd)
    return {**report, "signal_actions": actions}


def subenv(**kw):
    return {**os.environ, "PYTHONUNBUFFERED": "1", "VERL_API_DATA_DIR": str(ROOT), **kw}


class DB:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite3"
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""CREATE TABLE IF NOT EXISTS objects(id TEXT PRIMARY KEY,kind TEXT,state TEXT,payload TEXT,created TEXT);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,ref TEXT,kind TEXT,ts TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS event_ref ON events(ref,seq);
            CREATE TABLE IF NOT EXISTS metrics(seq INTEGER PRIMARY KEY,ref TEXT,ts TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS metric_ref ON metrics(ref,seq);""")

    @contextlib.contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def add(self, kind, payload, state="ready", id=None):
        id = id or ident(kind)
        t = stamp()
        obj = {
            **payload,
            "id": id,
            "kind": kind,
            "state": state,
            "created_at": t,
            "updated_at": t,
        }
        with self.connect() as c:
            c.execute(
                "INSERT INTO objects VALUES(?,?,?,?,?)", (id, kind, state, js(obj), t)
            )
        return obj

    def get(self, id, kind=None):
        with self.connect() as c:
            row = c.execute(
                "SELECT payload,kind FROM objects WHERE id=?", (id,)
            ).fetchone()
        check(
            row and (not kind or row["kind"] == kind),
            f"对象不存在: {id}",
            "NOT_FOUND",
            404,
        )
        return json.loads(row["payload"])

    def update(self, id, **kw):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT payload FROM objects WHERE id=?", (id,)).fetchone()
            check(row, f"对象不存在:{id}", "NOT_FOUND", 404)
            obj = {**json.loads(row[0]), **kw, "updated_at": stamp()}
            c.execute(
                "UPDATE objects SET payload=?,state=? WHERE id=?",
                (js(obj), obj["state"], id),
            )
        return obj

    def all(self, kind):
        with self.connect() as c:
            rows = c.execute(
                "SELECT payload FROM objects WHERE kind=? ORDER BY created DESC",
                (kind,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def emit(self, ref, kind, payload):
        with self.connect() as c:
            t = stamp()
            cur = c.execute(
                "INSERT INTO events(ref,kind,ts,payload) VALUES(?,?,?,?)",
                (ref, kind, t, js(payload)),
            )
            seq = cur.lastrowid
            if kind == "metric":
                c.execute(
                    "INSERT INTO metrics VALUES(?,?,?,?)", (seq, ref, t, js(payload))
                )
            if seq % 500 == 0:
                c.execute("DELETE FROM events WHERE seq<?", (max(0, seq - 50000),))
        return seq

    def events(self, after=0, ref=None, limit=200):
        where = "seq>?" + (" AND ref=?" if ref else "")
        args = [after] + ([ref] if ref else []) + [limit]
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM events WHERE " + where + " ORDER BY seq LIMIT ?", args
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "ref_id": r["ref"],
                "kind": r["kind"],
                "timestamp": r["ts"],
                "payload": json.loads(r["payload"]),
            }
            for r in rows
        ]

    def metric_rows(self, ref, after, limit):
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM metrics WHERE ref=? AND seq>? ORDER BY seq LIMIT ?",
                (ref, after, limit),
            ).fetchall()
        return [
            {"seq": r["seq"], "timestamp": r["ts"], **json.loads(r["payload"])}
            for r in rows
        ]


def state(db, id, value, **kw):
    obj = db.update(id, state=value, **kw)
    db.emit(id, "state", {"state": value, **kw})
    return obj


def settings(db):
    p = db.root / "settings.json"
    return {**SETTINGS, **(json.loads(p.read_text()) if p.exists() else {})}


def page(rows, offset=0, limit=50):
    return {
        "items": rows[offset : offset + limit],
        "total": len(rows),
        "offset": offset,
        "limit": limit,
    }


def public(job):
    obj = {k: v for k, v in job.items() if k not in ("internal_url", "internal_key")}
    if obj.get("command"):
        args = list(obj["command"])
        if "--api-key" in args:
            args[args.index("--api-key") + 1] = "<internal>"
        obj["command"] = args
    return obj


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class DatasetRequest(Schema):
    name: str = Field(min_length=1, max_length=200)
    task_type: Literal["cpt", "sft"] = "sft"


class DemoRequest(Schema):
    task_type: Literal["cpt", "sft"] = "sft"


class SamplesRequest(Schema):
    samples: list[dict[str, Any]] = Field(min_length=1, max_length=10000)


class EditRequest(Schema):
    sample: dict[str, Any]


class ImportRequest(Schema):
    path: str


class PrepareRequest(Schema):
    model_id: str = "base"
    max_length: int = Field(
        1024,
        ge=32,
        le=131072,
        description="上下文和答案总token上限；改变此值应重新准备数据",
    )
    validation_ratio: float = Field(0.05, ge=0, le=0.5)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    deduplicate: bool = True
    assistant_scope: Literal["all", "last"] = "all"
    overlength: Literal["error", "drop", "truncate"] = "error"
    on_invalid: Literal["error", "skip"] = "error"


class Lora(Schema):
    # 通常先用16/32；任务复杂或数据较多可试32/64、64/128，同时观察验证loss。
    # rank提高会增加可训练参数、梯度和优化器占用；不保证效果一定提高。
    rank: int = Field(16, ge=1, le=512, description="低秩维度；常用8/16/32/64，默认16")
    alpha: int = Field(
        32, ge=1, le=4096, description="标准LoRA缩放alpha/r，默认2；导出必须保留此值"
    )
    # all-linear覆盖注意力和MLP线性层；较省资源可指定q_proj/v_proj，
    # 或q_proj/k_proj/v_proj/o_proj。vLLM推理支持的最大rank由实例另设。
    target_modules: str | list[str] = Field(
        "all-linear", description="all-linear，或模块名数组/PEFT正则"
    )
    exclude_modules: str | list[str] | None = None


class TrainRequest(Schema):
    task_type: Literal["cpt", "sft"] = "sft"
    finetune: Literal["full", "lora"] = "lora"
    dataset_version_id: str
    model_id: str = "base"
    name: str = Field("training", max_length=200)
    gpu_ids: list[int | str] = Field(
        default_factory=lambda: [0, 1, 2, 3], min_length=1, max_length=64
    )
    # global batch = DP卡数 × 每卡micro batch × 梯度累积次数（静态batch模式）。
    # 4卡batch4/micro1是低门槛默认；增大global batch不必同时增大micro。
    train_batch_size: int = Field(4, ge=1, description="全局有效batch；需整除DP卡数")
    micro_batch_size_per_gpu: int = Field(
        1, ge=1, description="每卡一次前后向样本数；OOM先减此值或序列长度"
    )
    epochs: int = Field(1, ge=1, le=100000)
    max_steps: int | None = Field(None, ge=1)
    learning_rate: float | None = Field(
        None, gt=0, le=1, description="null：full=1e-5，LoRA=1e-4"
    )
    lora: Lora = Field(default_factory=Lora)
    save_freq: int = Field(-1, ge=-1, description="每N个优化步保存；-1仅末步，0不允许")
    test_freq: int = Field(
        -1, ge=-1, description="每N步验证；-1仅末步（有验证集时），0不允许"
    )
    max_checkpoints: int | None = Field(
        None, ge=1, description="每任务保留最近N份分片；null全保留，注意磁盘"
    )
    seed: int = Field(42, ge=0, le=2**31 - 1)
    verl_overrides: dict[str, Any] = Field(
        default_factory=dict, description="扁平原生配置；覆盖常用字段，管理字段除外"
    )
    runtime_env: dict[str, str] = Field(default_factory=dict)


class ResumeRequest(Schema):
    checkpoint_id: str
    additional_epochs: int = Field(1, ge=1, le=100000)


class ExportRequest(Schema):
    checkpoint_id: str
    name: str = Field("exported-model", min_length=1, max_length=200)
    format: Literal["auto", "adapter", "merged"] = "auto"


class RegisterRequest(Schema):
    name: str = Field(min_length=1, max_length=200)
    path: str
    model_kind: Literal["full", "lora"] = "full"
    base_model_id: str | None = None


class ChatOptions(Schema):
    enable_thinking: bool | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=-1)
    min_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    repetition_penalty: float | None = Field(None, gt=0, le=3)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    stop: str | list[str] | None = None
    thinking_token_budget: int | None = Field(None, ge=1, le=131072)


class Message(Schema):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(ChatOptions):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=1000)
    stream: bool = True
    instance_id: str | None = None


class InferenceRequest(Schema):
    model_id: str = "base"
    gpu_ids: list[int | str] = Field(
        default_factory=lambda: [0], min_length=1, max_length=64
    )
    tensor_parallel_size: int = Field(1, ge=1, le=64)
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    gpu_memory_utilization: float = Field(0.7, gt=0, lt=1)
    max_model_len: int = Field(8192, ge=128, le=131072)
    max_num_seqs: int = Field(2, ge=1, le=1024)
    max_lora_rank: int = Field(64, ge=1, le=512)
    enforce_eager: bool = True
    enable_prefix_caching: bool = True
    extra_args: dict[str, Any] = Field(default_factory=dict)


class AdapterRequest(Schema):
    model_id: str


class SettingsRequest(Schema):
    cors_origins: list[str] | None = None
    gpu_poll_seconds: float | None = Field(None, ge=0.5, le=30)
    auto_unload_inference: bool | None = None
    max_cpu_jobs: int | None = Field(None, ge=1, le=4)
    max_upload_bytes: int | None = Field(None, ge=1024, le=100 * 1024**3)


def tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )


def fingerprint(path):
    p = Path(path)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "special_tokens_map.json",
    )
    return {n: sha(p / n) for n in names if (p / n).is_file()}


def normalize(row, task):
    check(isinstance(row, dict), "样本需JSON对象", "INVALID_SAMPLE", 422)
    meta = {
        "source_id": str(row.get("source_id") or ""),
        "metadata": row.get("metadata", {}),
    }
    if task == "cpt":
        check(
            isinstance(row.get("text"), str) and row["text"].strip(),
            "CPT text需非空字符串",
            "INVALID_SAMPLE",
            422,
        )
        return {"text": row["text"], **meta}
    msgs = row.get("messages")
    if msgs is None and "instruction" in row:
        check(
            all(
                isinstance(row.get(k, ""), str)
                for k in ("instruction", "input", "output")
            ),
            "instruction/input/output必须字符串",
            "INVALID_SAMPLE",
            422,
        )
        msgs = [
            {
                "role": "user",
                "content": row["instruction"]
                + ("\n\n" + row["input"] if row.get("input") else ""),
            },
            {"role": "assistant", "content": row.get("output", "")},
        ]
    check(
        isinstance(msgs, list) and len(msgs) >= 2,
        "SFT需messages",
        "INVALID_SAMPLE",
        422,
    )
    copied = []
    for i, msg in enumerate(msgs):
        check(
            isinstance(msg, dict)
            and msg.get("role") in ("system", "user", "assistant"),
            f"消息{i}角色错误",
            "INVALID_SAMPLE",
            422,
        )
        check(
            isinstance(msg.get("content"), str) and msg["content"].strip(),
            f"消息{i}内容为空",
            "INVALID_SAMPLE",
            422,
        )
        check(
            not (set(msg) - {"role", "content", "reasoning_content"}),
            f"消息{i}包含未支持字段",
            "INVALID_SAMPLE",
            422,
        )
        if "reasoning_content" in msg:
            check(
                msg["role"] == "assistant"
                and isinstance(msg["reasoning_content"], str),
                "reasoning_content需assistant字符串",
                "INVALID_SAMPLE",
                422,
            )
        copied.append(dict(msg))
    roles = [m["role"] for m in copied]
    core = roles[1:] if roles[0] == "system" else roles
    check(
        len(core) % 2 == 0
        and core == [x for _ in range(len(core) // 2) for x in ("user", "assistant")],
        "需system?+user/assistant交替，末尾assistant",
        "INVALID_SAMPLE",
        422,
    )
    if "enable_thinking" in row:
        check(
            type(row["enable_thinking"]) is bool,
            "enable_thinking需bool",
            "INVALID_SAMPLE",
            422,
        )
        meta["enable_thinking"] = row["enable_thinking"]
    return {"messages": copied, **meta}


def rows(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        yield {"text": path.read_text(encoding="utf-8-sig"), "source_id": path.name}
    elif ext == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as e:
                        raise Error(f"JSONL第{i}行:{e}", "INVALID_JSONL", 422)
    elif ext == ".json":
        x = json.loads(path.read_text(encoding="utf-8-sig"))
        x = x.get("samples") if isinstance(x, dict) else x
        check(isinstance(x, list), "JSON需数组或samples数组", "INVALID_DATA", 422)
        yield from x
    elif ext == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise Error("仅支持txt/md/json/jsonl/parquet", "UNSUPPORTED_FORMAT", 422)


@contextlib.contextmanager
def dslock(db, id):
    p = db.root / "datasets" / id
    p.mkdir(parents=True, exist_ok=True)
    with (p / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield p


def append(db, id, records):
    ds = db.get(id, "dataset")
    with dslock(db, id) as p:
        temp = p / (ident("append") + ".jsonl")
        count = 0
        try:
            with temp.open("w", encoding="utf-8") as out:
                for row in records:
                    out.write(js(normalize(row, ds["task_type"])) + "\n")
                    count += 1
            with (p / "raw.jsonl").open("ab") as out, temp.open("rb") as src:
                shutil.copyfileobj(src, out)
            return db.update(id, sample_count=db.get(id)["sample_count"] + count)
        finally:
            temp.unlink(missing_ok=True)


def sft_tokens(tok, row, scope):
    # Freeze the selected template for the full/prefix pair. In Transformers 5,
    # apply_chat_template defaults to BatchEncoding: len(result) counts fields,
    # not tokens. Always request a flat list explicitly, in both TF4 and TF5.
    template = tok.get_chat_template()
    targets = [i for i, m in enumerate(row["messages"]) if m["role"] == "assistant"]
    if scope == "last":
        targets = targets[-1:]
    for i in targets:
        chat = []
        for msg in row["messages"][: i + 1]:
            msg = dict(msg)
            reason = msg.pop("reasoning_content", "")
            if reason:
                check(
                    "<think>" not in msg["content"],
                    "思考块不能重复",
                    "INVALID_SAMPLE",
                    422,
                )
                msg["content"] = (
                    "<think>\n" + reason + "\n</think>\n\n" + msg["content"]
                )
            chat.append(msg)
        reason = bool(re.search(r"<think>\s*\S[\s\S]*?</think>", chat[-1]["content"]))
        thinking = row.get("enable_thinking", reason)
        check(
            thinking == reason,
            "thinking=true需完整非空思考示范；false不能含非空思考块",
            "INVALID_SAMPLE",
            422,
        )
        kw = {
            "chat_template": template,
            "enable_thinking": thinking,
            "return_dict": False,
            "return_tensors": None,
            "padding": False,
            "truncation": False,
        }
        prefix_text = tok.apply_chat_template(
            chat[:-1], add_generation_prompt=True, tokenize=False, **kw
        )
        full_text = tok.apply_chat_template(
            chat, add_generation_prompt=False, tokenize=False, **kw
        )
        prefix_ids = tok.apply_chat_template(
            chat[:-1], add_generation_prompt=True, tokenize=True, **kw
        )
        full_ids = tok.apply_chat_template(
            chat, add_generation_prompt=False, tokenize=True, **kw
        )
        check(
            all(isinstance(ids, list) and all(type(t) is int for t in ids)
                for ids in (prefix_ids, full_ids)),
            "模板token化结果必须是一维整数列表",
            "TEMPLATE_MISMATCH",
            422,
        )
        # apply_chat_template(tokenize=True) internally uses add_special_tokens=False.
        # Verify that rendering and tokenization agree; never concatenate tokens
        # or silently trim a prefix to make a mismatching template pass.
        check(
            prefix_ids == tok.encode(prefix_text, add_special_tokens=False)
            and full_ids == tok.encode(full_text, add_special_tokens=False),
            "模板文本与实际token化结果不一致",
            "TEMPLATE_MISMATCH",
            422,
        )
        check(
            full_text.startswith(prefix_text),
            "完整模板与生成前缀文本不一致，请检查模型模板",
            "TEMPLATE_MISMATCH",
            422,
        )
        check(
            len(full_ids) > len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids,
            "完整模板与生成前缀token不一致，请检查模型模板",
            "TEMPLATE_MISMATCH",
            422,
        )
        yield {
            "input_ids": full_ids,
            "loss_mask": [0] * len(prefix_ids) + [1] * (len(full_ids) - len(prefix_ids)),
            "target_turn": i,
            "enable_thinking": thinking,
        }


def cpt_tokens(tok, row, length):
    ids = tok.encode(row["text"], add_special_tokens=False)
    check(tok.eos_token_id is not None, "CPT需要EOS")
    if not ids or ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    start = 0
    while start < len(ids) - 1:
        part = ids[start : start + length]
        yield {
            "input_ids": part,
            "loss_mask": [0] + [1] * (len(part) - 1),
            "target_turn": -1,
            "enable_thinking": False,
        }
        start += len(part) - 1


class PreparedDataset:
    """verl custom_cls。mask不在这里shift，首token为0防止样本边界串监督。"""

    def __init__(
        self, parquet_files, tokenizer, config, processor=None, max_samples=-1
    ):
        import pandas as pd

        check(config.get("pad_mode", "no_padding") == "no_padding", "仅支持no_padding")
        files = (
            [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        )
        self.data = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if max_samples > 0:
            self.data = self.data.iloc[:max_samples]
        self.max_length = int(config.max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        import torch

        r = self.data.iloc[i]
        ids = torch.tensor(list(r.input_ids), dtype=torch.long)
        mask = torch.tensor(list(r.loss_mask), dtype=torch.long)
        check(
            len(ids) == len(mask) and len(ids) <= self.max_length and int(mask[0]) == 0,
            "准备数据/mask不匹配",
        )
        return {
            "input_ids": ids,
            "loss_mask": mask,
            "position_ids": torch.arange(len(ids), dtype=torch.long),
        }


def prepare_data(db, job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = PrepareRequest.model_validate(job["request"])
    ds = db.get(job["dataset_id"], "dataset")
    model = db.get(cfg.model_id, "model")
    tp = model.get("tokenizer_path", model["path"])
    tok = tokenizer(tp)
    fp = fingerprint(tp)
    out = db.root / "dataset_versions" / job["version_id"]
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_samples": 0,
        "duplicates": 0,
        "invalid": 0,
        "dropped": 0,
        "truncated": 0,
        "train_samples": 0,
        "val_samples": 0,
        "total_tokens": 0,
        "supervised_tokens": 0,
        "issues": [],
        "warnings": [],
    }
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int64())),
            ("source_id", pa.string()),
            ("raw_index", pa.int64()),
            ("target_turn", pa.int64()),
            ("enable_thinking", pa.bool_()),
        ]
    )
    seen = set()
    lengths = []
    buffers = {s: [] for s in ("train", "val")}
    writers = {
        s: pq.ParquetWriter(out / f"{s}.parquet", schema, compression="zstd")
        for s in buffers
    }
    try:
        for i, raw in enumerate(rows(job["snapshot"])):
            report["raw_samples"] += 1
            try:
                r = normalize(raw, ds["task_type"])
                content = {
                    k: v for k, v in r.items() if k not in ("source_id", "metadata")
                }
                digest = hashlib.sha256(js(content).encode()).hexdigest()
                if cfg.deduplicate and digest in seen:
                    report["duplicates"] += 1
                    continue
                seen.add(digest)
                source = r["source_id"] or digest
                score = (
                    int(
                        hashlib.sha256(f"{cfg.seed}:{source}".encode()).hexdigest()[
                            :16
                        ],
                        16,
                    )
                    / 2**64
                )
                split = "val" if score < cfg.validation_ratio else "train"
                # 每条原始样本先整体校验，skip不会留下半条多轮对话。
                encoded = list(
                    cpt_tokens(tok, r, cfg.max_length)
                    if ds["task_type"] == "cpt"
                    else sft_tokens(tok, r, cfg.assistant_scope)
                )
                accepted = []
                for item in encoded:
                    if len(item["input_ids"]) > cfg.max_length:
                        if cfg.overlength == "error":
                            raise Error(
                                f"样本{i}长度{len(item['input_ids'])}>{cfg.max_length}",
                                "TOO_LONG",
                                422,
                            )
                        if cfg.overlength == "drop":
                            report["dropped"] += 1
                            continue
                        item["input_ids"] = item["input_ids"][: cfg.max_length]
                        item["loss_mask"] = item["loss_mask"][: cfg.max_length]
                        report["truncated"] += 1
                    check(
                        sum(item["loss_mask"]) > 0,
                        "没有有效监督token",
                        "NO_TARGET_TOKENS",
                        422,
                    )
                    accepted.append({**item, "source_id": source, "raw_index": i})
                for item in accepted:
                    buffers[split].append(item)
                    report[split + "_samples"] += 1
                    report["total_tokens"] += len(item["input_ids"])
                    report["supervised_tokens"] += sum(item["loss_mask"])
                    lengths.append(len(item["input_ids"]))
                    if len(buffers[split]) >= 256:
                        writers[split].write_table(
                            pa.Table.from_pylist(buffers[split], schema=schema)
                        )
                        buffers[split].clear()
            except Exception as e:
                if isinstance(e, Error) and e.code == "TEMPLATE_MISMATCH":
                    # Template/encoding bugs are fatal even when invalid data may
                    # be skipped. Preserve the response schema and locate the row.
                    e.message = f"样本index={i}, source_id={raw.get('source_id', '')!r}: {e}"
                    e.args = (e.message,)
                report["invalid"] += 1
                if len(report["issues"]) < 100:
                    report["issues"].append({"raw_index": i, "error": str(e)})
                if cfg.on_invalid == "error" or getattr(e, "code", None) == "TEMPLATE_MISMATCH":
                    raise
            if i % 100 == 0:
                db.update(job["id"], progress={"processed_samples": i + 1})
        for s in buffers:
            if buffers[s]:
                writers[s].write_table(pa.Table.from_pylist(buffers[s], schema=schema))
    finally:
        for w in writers.values():
            w.close()
        atomic(out / "report.json", report)
    check(
        report["train_samples"] > 0,
        "训练集为空，减少验证比例或增加样本",
        "EMPTY_DATASET",
        422,
    )
    if cfg.validation_ratio and not report["val_samples"]:
        report["warnings"].append("分组后验证集为空，请增加来源组或调整seed")
    lengths.sort()
    report["lengths"] = {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / len(lengths),
        "p95": lengths[int((len(lengths) - 1) * 0.95)],
    }
    atomic(out / "report.json", report)
    manifest = {
        "dataset_id": ds["id"],
        "task_type": ds["task_type"],
        "model_id": cfg.model_id,
        "tokenizer_path": tp,
        "fingerprint": fp,
        "prepare_config": cfg.model_dump(),
        "report": report,
        "path": str(out),
        "raw_sha256": sha(job["snapshot"]),
        "train_sha256": sha(out / "train.parquet"),
        "val_sha256": sha(out / "val.parquet"),
    }
    atomic(out / "manifest.json", manifest)
    db.add("dataset_version", manifest, id=job["version_id"])
    return {"dataset_version_id": job["version_id"], "report": report}


def gpu_snapshot(db):
    fields = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.sm,driver_version"

    def num(v, scale=1):
        try:
            return float(v.strip()) * scale
        except ValueError:
            return None

    try:
        output = command(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            timeout=8,
        )
        gpus = []
        for r in csv.reader(io.StringIO(output)):
            r = [v.strip() for v in r]
            if len(r) != 13:
                continue
            gpus.append(
                dict(
                    index=int(r[0]),
                    uuid=r[1],
                    name=r[2],
                    memory_total_bytes=num(r[3], 1024**2),
                    memory_used_bytes=num(r[4], 1024**2),
                    memory_free_bytes=num(r[5], 1024**2),
                    utilization_gpu_percent=num(r[6]),
                    utilization_memory_percent=num(r[7]),
                    temperature_c=num(r[8]),
                    power_w=num(r[9]),
                    power_limit_w=num(r[10]),
                    sm_clock_mhz=num(r[11]),
                    driver_version=r[12],
                    processes=[],
                    reserved_by=[],
                )
            )
        err = None
        try:
            procs = command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                timeout=8,
            )
            for r in csv.reader(io.StringIO(procs)):
                if len(r) < 4:
                    continue
                for g in gpus:
                    if g["uuid"] == r[0].strip():
                        g["processes"].append(
                            {
                                "pid": int(r[1]),
                                "name": r[2].strip(),
                                "memory_used_bytes": num(r[-1], 1024**2),
                            }
                        )
        except Exception as e:
            err = str(e)
        for j in db.all("job"):
            if j["state"] in LIVE:
                for g in gpus:
                    if g["uuid"] in j.get("gpu_uuids", []):
                        g["reserved_by"].append(
                            {
                                "job_id": j["id"],
                                "job_type": j["job_type"],
                                "state": j["state"],
                            }
                        )
        return {
            "available": True,
            "timestamp": stamp(),
            "gpus": gpus,
            "process_query_error": err,
        }
    except Exception as e:
        return {"available": False, "timestamp": stamp(), "gpus": [], "error": str(e)}


def select_gpus(ids, snapshot):
    check(
        snapshot["available"],
        snapshot.get("error", "GPU不可用"),
        "GPU_UNAVAILABLE",
        503,
    )
    result = []
    for i in ids:
        found = [
            g for g in snapshot["gpus"] if str(i) == str(g["index"]) or i == g["uuid"]
        ]
        check(len(found) == 1, f"未知GPU:{i}", "UNKNOWN_GPU", 422)
        result.append(found[0]["uuid"])
    check(len(set(result)) == len(result), "GPU不能重复", "INVALID_GPU", 422)
    return result


def verl_path():
    s = importlib.util.find_spec("verl")
    check(
        s and s.submodule_search_locations,
        "当前Python未安装verl",
        "DEPENDENCY_MISSING",
        503,
    )
    return Path(list(s.submodule_search_locations)[0])


def compose(overrides=None):
    from hydra import compose as hc
    from hydra import initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    with initialize_config_dir(
        config_dir=str(verl_path() / "trainer" / "config"), version_base=None
    ):
        cfg = hc(config_name="sft_trainer_engine")
    with open_dict(cfg):
        for k, v in (overrides or {}).items():
            OmegaConf.update(cfg, k, v, merge=False, force_add=True)
    return cfg


def probe():
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(compose(), resolve=False)
    help_text = ""
    budget = False
    commit = None
    if version("vllm"):
        with contextlib.suppress(Exception):
            help_text = command(
                [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                timeout=90,
            )
        try:
            import inspect

            from vllm import SamplingParams

            budget = (
                "thinking_token_budget" in inspect.signature(SamplingParams).parameters
            )
        except Exception:
            pass
    with contextlib.suppress(Exception):
        commit = command(
            ["git", "-C", str(verl_path().parent), "rev-parse", "HEAD"], timeout=5
        ).strip()
    return {
        "defaults": cfg,
        "vllm_help": help_text,
        "thinking_token_budget": budget,
        "verl_commit": commit,
        "versions": {
            n: version(n) for n in ("verl", "torch", "vllm", "transformers", "peft")
        },
    }


ENV_ALLOW = {
    "OMP_NUM_THREADS",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SHM_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_LEVEL",
}
PROTECTED = {
    "model.path",
    "model.tokenizer_path",
    "model.lora_adapter_path",
    "model.external_lib",
    "model.trust_remote_code",
    "data.train_files",
    "data.val_files",
    "data.custom_cls",
    "data.pad_mode",
    "data.max_length",
    "trainer.default_local_dir",
    "trainer.default_hdfs_dir",
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.logger",
    "trainer.nnodes",
    "trainer.n_gpus_per_node",
    "trainer.device",
    "checkpoint.save_contents",
    "checkpoint.load_contents",
}


def key_exists(obj, key):
    for k in key.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return False
        obj = obj[k]
    return True


def advanced(values, defaults):
    def dynamic(v):
        if isinstance(v, dict):
            return any(str(k).startswith("_") or dynamic(x) for k, x in v.items())
        if isinstance(v, list):
            return any(dynamic(x) for x in v)
        return isinstance(v, str) and "${" in v

    for k, v in values.items():
        check(
            re.fullmatch(
                r"(?:model|engine|data|optim|trainer|checkpoint)\.[A-Za-z0-9_.]+", k
            ),
            f"不允许配置键:{k}",
        )
        check(
            not any(
                k == p or k.startswith(p + ".") or p.startswith(k + ".")
                for p in PROTECTED
            ),
            f"{k}由服务管理",
        )
        check("_target_" not in k and not dynamic(v), "不接受动态类/插值")
        check(key_exists(defaults, k), f"当前verl没有配置项:{k}")
        check(not k.startswith("model.lora."), "当前FSDP仅支持扁平LoRA配置")


def training_config(db, req, output, environment, resume=None):
    r = TrainRequest.model_validate(req)
    dv = db.get(r.dataset_version_id, "dataset_version")
    model = db.get(r.model_id, "model")
    base = (
        db.get(model["base_model_id"], "model")
        if model["model_kind"] == "lora"
        else model
    )
    check(dv["task_type"] == r.task_type, "数据集与训练类型不一致")
    check(
        model["model_kind"] != "lora" or r.finetune == "lora",
        "adapter继续训练需finetune=lora，全参请先合并",
    )
    check(
        fingerprint(base.get("tokenizer_path", base["path"])) == dv["fingerprint"],
        "tokenizer或模板改变，请重新prepare",
    )
    check(not (set(r.runtime_env) - ENV_ALLOW), "runtime_env包含未支持键")
    check(r.save_freq != 0 and r.test_freq != 0, "save_freq/test_freq不能为0")
    advanced(r.verl_overrides, environment["defaults"])
    p = Path(dv["path"])
    n = len(r.gpu_ids)
    v = {
        "model.path": base["path"],
        "model.enable_gradient_checkpointing": True,
        "model.use_remove_padding": True,
        "model.lora_rank": r.lora.rank if r.finetune == "lora" else 0,
        "model.lora_alpha": r.lora.alpha,
        "model.target_modules": r.lora.target_modules,
        "model.exclude_modules": r.lora.exclude_modules,
        "model.lora_adapter_path": model["path"]
        if model["model_kind"] == "lora"
        else None,
        "engine.strategy": "fsdp2",
        "engine.model_dtype": "fp32",
        "engine.use_torch_compile": False,
        "engine.offload_policy": False,
        "engine.ulysses_sequence_parallel_size": 1,
        "data.train_files": str(p / "train.parquet"),
        "data.val_files": str(p / "val.parquet")
        if dv["report"]["val_samples"]
        else None,
        "data.train_batch_size": r.train_batch_size,
        "data.micro_batch_size_per_gpu": r.micro_batch_size_per_gpu,
        "data.use_dynamic_bsz": False,
        "data.pad_mode": "no_padding",
        "data.max_length": dv["prepare_config"]["max_length"],
        "data.num_workers": 0,
        "data.custom_cls.path": str(SCRIPT),
        "data.custom_cls.name": "PreparedDataset",
        "optim.lr": r.learning_rate or (1e-4 if r.finetune == "lora" else 1e-5),
        "optim.override_optimizer_config": {"foreach": False},
        "trainer.total_epochs": r.epochs,
        "trainer.total_training_steps": r.max_steps,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": n,
        "trainer.logger": ["console"],
        "trainer.resume_mode": "resume_path" if resume else "disable",
        "trainer.resume_from_path": resume,
        "trainer.default_local_dir": str(output),
        "trainer.project_name": "verl-api",
        "trainer.experiment_name": r.name,
        "trainer.save_freq": r.save_freq,
        "trainer.test_freq": r.test_freq,
        "trainer.max_ckpt_to_keep": r.max_checkpoints,
        "trainer.seed": r.seed,
        "checkpoint.save_contents": ["model", "optimizer", "extra"],
    }
    v.update(r.verl_overrides)
    for key in (
        "data.train_batch_size",
        "data.micro_batch_size_per_gpu",
        "engine.ulysses_sequence_parallel_size",
        "trainer.total_epochs",
        "model.lora_rank",
        "model.lora_alpha",
    ):
        check(type(v[key]) is int, f"{key}必须是整数")
    check(
        v["model.lora_rank"] >= 0 and v["model.lora_alpha"] > 0, "LoRA rank/alpha不合法"
    )
    check(
        type(v["optim.lr"]) in (int, float) and 0 < v["optim.lr"] <= 1,
        "optim.lr需在(0,1]内",
    )
    for key in ("trainer.save_freq", "trainer.test_freq"):
        check(
            v[key] == "after_each_epoch"
            or (type(v[key]) is int and v[key] != 0 and v[key] >= -1),
            f"{key}需为-1、正整数或after_each_epoch",
        )
    check(
        v["data.val_files"] or v["trainer.test_freq"] == -1,
        "没有验证集时test_freq只能为-1；否则verl会尝试遍历空验证集",
    )
    check(v["engine.strategy"] in ("fsdp", "fsdp2"), "本版只适配FSDP/FSDP2")
    check(
        (int(v["model.lora_rank"]) > 0) == (r.finetune == "lora"),
        "lora_rank和finetune冲突",
    )
    if model["model_kind"] == "lora":
        ac = json.loads((Path(model["path"]) / "adapter_config.json").read_text())
        v.update(
            {
                "model.lora_rank": ac["r"],
                "model.lora_alpha": ac["lora_alpha"],
                "model.target_modules": ac["target_modules"],
            }
        )
    batch = int(v["data.train_batch_size"])
    micro = int(v["data.micro_batch_size_per_gpu"])
    sp = int(v["engine.ulysses_sequence_parallel_size"])
    check(sp > 0 and n % sp == 0, "GPU数与序列并行不兼容")
    dp = n // sp
    check(batch > 0 and batch % dp == 0, "global batch需是DP大小的整数倍")
    check(
        micro > 0 and (v["data.use_dynamic_bsz"] or (batch // dp) % micro == 0),
        "micro batch需整除每DP batch",
    )
    count = dv["report"]["train_samples"]
    max_samples = v.get("data.train_max_samples", -1)
    if max_samples > 0:
        count = min(count, max_samples)
    per_epoch = count // batch
    check(per_epoch > 0, "训练集不足一个完整global batch")
    if v["data.val_files"]:
        check(
            dv["report"]["val_samples"] >= batch,
            "验证集少于global batch，verl会drop_last；请调整数据/批量",
        )
    epochs = int(v["trainer.total_epochs"])
    steps = v["trainer.total_training_steps"]
    check(epochs > 0 and (steps is None or int(steps) > 0), "epochs/steps必须为正")
    possible = epochs * per_epoch
    warnings = []
    if steps and steps > possible:
        warnings.append(f"max_steps={steps}超过epochs上限，实际最多{possible}步")
    v["trainer.total_training_steps"] = min(steps, possible) if steps else None
    return {
        "overrides": v,
        "base_model_id": base["id"],
        "steps_per_epoch": per_epoch,
        "expected_total_steps": min(steps, possible) if steps else possible,
        "warnings": warnings,
        "data_parallel_size": dp,
    }


def verify_model(path, kind):
    p = Path(path)
    check(p.is_dir(), f"模型目录不存在:{p}")
    if kind == "lora":
        check(
            (p / "adapter_config.json").is_file()
            and any(
                (p / n).is_file()
                for n in ("adapter_model.safetensors", "adapter_model.bin")
            ),
            "adapter配置或权重缺失",
        )
        ac = json.loads((p / "adapter_config.json").read_text())
        check(
            int(ac.get("r", 0)) > 0 and float(ac.get("lora_alpha", 0)) > 0,
            "adapter r/alpha需>0",
        )
    else:
        check(
            (p / "config.json").is_file()
            and (any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))),
            "缺少完整HF模型配置/权重",
        )
        for index in p.glob("*.index.json"):
            for name in set(
                json.loads(index.read_text()).get("weight_map", {}).values()
            ):
                check((p / name).is_file(), f"权重分片缺失:{name}")


def checkpoint(db, job, step):
    p = Path(job["output_dir"]) / f"global_step_{step}"
    n = len(job["gpu_uuids"])
    check(
        all((p / f"model_world_size_{n}_rank_{r}.pt").is_file() for r in range(n)),
        "保存回调完成但checkpoint模型分片不齐",
        "CHECKPOINT_INCOMPLETE",
        500,
    )
    v = job["training_config"]["overrides"]
    data = {
        "job_id": job["id"],
        "step": step,
        "path": str(p),
        "world_size": n,
        "finetune": job["request"]["finetune"],
        "model_id": job["request"]["model_id"],
        "base_model_id": job["training_config"]["base_model_id"],
        "dataset_version_id": job["request"]["dataset_version_id"],
        "lora_rank": v["model.lora_rank"],
        "lora_alpha": v["model.lora_alpha"],
        "resumable": all(
            (p / f"optim_world_size_{n}_rank_{r}.pt").is_file()
            and (p / f"extra_state_world_size_{n}_rank_{r}.pt").is_file()
            for r in range(n)
        )
        and all(
            (p / f"data_{r}.pt").is_file()
            for r in range(job["training_config"]["data_parallel_size"])
        ),
    }
    atomic(p / "verl_api_complete.json", data)
    cid = f"ckpt_{job['id']}_{step}"
    try:
        db.get(cid)
        db.update(cid, **data)
    except Error:
        db.add("checkpoint", data, id=cid)
    for old in db.all("checkpoint"):
        if old["job_id"] == job["id"] and not Path(old["path"]).is_dir():
            db.update(old["id"], state="missing", resumable=False)
    db.emit(job["id"], "checkpoint", {"checkpoint_id": cid, **data})
    return cid


def metric_summary(data):
    aliases = {
        "loss": ["train/loss", "train_loss", "loss"],
        "validation_loss": ["val/loss", "eval/loss"],
        "learning_rate": ["train/lr", "lr", "learning_rate"],
        "grad_norm": ["train/grad_norm", "grad_norm"],
        "epoch": ["train/epoch", "epoch"],
        "tokens_per_second": ["api/tokens_per_second", "train/tokens_per_second"],
        "step_seconds": ["api/step_wall_seconds", "train/step_time"],
    }
    return {
        name: next(
            (data[k] for k in keys if isinstance(data.get(k), (int, float))), None
        )
        for name, keys in aliases.items()
    }


def train_rank():
    import torch.distributed as dist
    from verl.trainer import sft_trainer
    from verl.utils.tracking import Tracking

    db = DB()
    job = db.get(os.environ["VERL_API_JOB"], "job")
    rank = int(os.environ.get("RANK", "0"))
    os.environ.update(job["request"].get("runtime_env", {}))
    started = time.monotonic()
    clock = [started, -1]
    original_log = Tracking.log
    original_init = sft_trainer.SFTTrainer.__init__

    def log(self, data, *args, **kw):
        result = original_log(self, data, *args, **kw)
        if rank == 0:
            step = int(kw.get("step", args[0] if args else 0))
            values = clean(data)
            tick = time.monotonic()
            if "train/loss" in values and step > clock[1]:
                wall = tick - clock[0]
                values["api/step_wall_seconds"] = wall
                if isinstance(values.get("train/global_tokens"), (int, float)):
                    values["api/tokens_per_second"] = values[
                        "train/global_tokens"
                    ] / max(wall, 1e-6)
                clock[:] = [tick, step]
            summary = metric_summary(values)
            if summary["epoch"] is None:
                summary["epoch"] = step / job["training_config"]["steps_per_epoch"]
            payload = {
                "step": step,
                "elapsed_seconds": tick - started,
                "metrics": values,
                "summary": summary,
            }
            db.emit(job["id"], "metric", payload)
            old = db.get(job["id"]).get("progress", {}).get("last_metrics", {})
            db.update(
                job["id"],
                progress={
                    "step": step,
                    "total_steps": job["training_config"]["expected_total_steps"],
                    "epoch": summary["epoch"],
                    "last_metrics": {
                        **old,
                        **{k: v for k, v in summary.items() if v is not None},
                    },
                    "elapsed_seconds": tick - started,
                },
            )
        return result

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        clock[:] = [time.monotonic(), self.resume_global_step]
        save = self.ckpt_handler.save_checkpoint

        def wrapped(*a, **kw):
            result = save(*a, **kw)
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                checkpoint(db, job, int(kw.get("step", a[0] if a else 0)))
            return result

        self.ckpt_handler.save_checkpoint = wrapped

    Tracking.log = log
    sft_trainer.SFTTrainer.__init__ = init
    sft_trainer.run_sft(compose(job["training_config"]["overrides"]))


def logged_process(db, job, argv, env=None, cwd=None, ready=None):
    p = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    db.update(job["id"], child_pid=p.pid, child_pid_start=proc_start(p.pid), command=argv)

    def drain():
        with Path(job["log_path"]).open("a", encoding="utf-8", buffering=1) as out:
            for line in p.stdout:
                out.write(line)
                db.emit(job["id"], "log", {"text": line[:16000]})
        p.stdout.close()

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    deadline = time.monotonic() + 900
    isready = False
    while p.poll() is None:
        if ready and not isready:
            isready = ready()
            if not isready and time.monotonic() > deadline:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                raise Error("推理启动超过900秒，请查看日志", "INFERENCE_TIMEOUT", 503)
        time.sleep(0.5)
    t.join(timeout=10)
    check(
        p.returncode == 0,
        f"子进程退出码{p.returncode}，详见logs",
        "PROCESS_FAILED",
        500,
    )


def export_model(db, job):
    r = ExportRequest.model_validate(job["request"])
    ck = db.get(r.checkpoint_id, "checkpoint")
    p = Path(ck["path"])
    check((p / "verl_api_complete.json").is_file(), "检查点已删除或没有完整标记")
    lora = ck["finetune"] == "lora"
    check(lora or r.format != "adapter", "全参不能导出adapter")
    base = db.get(ck["base_model_id"], "model")
    out = Path(job["output_dir"])
    stage = out / "hf_stage"
    final = out / "model"
    kind = "full"
    if lora:
        atomic(
            p / "lora_train_meta.json",
            {
                "r": ck["lora_rank"],
                "lora_alpha": ck["lora_alpha"],
                "task_type": "CAUSAL_LM",
            },
        )
    logged_process(
        db,
        job,
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(p),
            "--target_dir",
            str(stage),
        ],
        cwd=str(verl_path().parent),
    )
    if lora:
        adapter = stage / "lora_adapter"
        check(
            (adapter / "adapter_config.json").is_file(),
            "当前merger没有生成adapter",
            "EXPORT_FAILED",
            500,
        )
        ac = json.loads((adapter / "adapter_config.json").read_text())
        check(int(ac["r"]) == ck["lora_rank"], "adapter rank不匹配")
        ac.update(
            lora_alpha=ck["lora_alpha"],
            base_model_name_or_path=base["path"],
            task_type="CAUSAL_LM",
            inference_mode=True,
        )
        atomic(adapter / "adapter_config.json", ac)
        if r.format == "merged":
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                base["path"],
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                local_files_only=True,
                trust_remote_code=False,
            )
            tuned = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
            merged = tuned.merge_and_unload()
            merged.save_pretrained(final, safe_serialization=True, max_shard_size="4GB")
            tokenizer(base.get("tokenizer_path", base["path"])).save_pretrained(final)
        else:
            kind = "lora"
            adapter.rename(final)
            tokenizer(base.get("tokenizer_path", base["path"])).save_pretrained(final)
        shutil.rmtree(stage)
    else:
        stage.rename(final)
    verify_model(final, kind)
    model = db.add(
        "model",
        {
            "name": r.name,
            "model_kind": kind,
            "path": str(final),
            "tokenizer_path": base.get("tokenizer_path", base["path"])
            if kind == "lora"
            else str(final),
            "base_model_id": base["id"] if kind == "lora" else None,
            "parent_model_id": ck["model_id"],
            "checkpoint_id": ck["id"],
            "step": ck["step"],
            "chat_defaults": {},
            "export_job_id": job["id"],
        },
    )
    return {"model_id": model["id"], "path": str(final), "model_kind": kind}


VLLM_EXTRA = {
    "max-num-batched-tokens",
    "kv-cache-dtype",
    "swap-space",
    "cpu-offload-gb",
    "disable-custom-all-reduce",
    "enable-chunked-prefill",
    "disable-log-requests",
}


def inference_worker(db, job):
    import httpx

    r = InferenceRequest.model_validate(job["request"])
    model = db.get(r.model_id, "model")
    base = (
        db.get(model["base_model_id"], "model")
        if model["model_kind"] == "lora"
        else model
    )
    environment = json.loads((db.root / "probe.json").read_text())
    help_text = environment["vllm_help"]
    check(help_text, "vLLM探测失败", "VLLM_UNAVAILABLE", 503)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{port}"
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base["path"],
        "--served-model-name",
        base["id"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        key,
        "--tensor-parallel-size",
        str(r.tensor_parallel_size),
        "--dtype",
        r.dtype,
        "--gpu-memory-utilization",
        str(r.gpu_memory_utilization),
        "--max-model-len",
        str(r.max_model_len),
        "--max-num-seqs",
        str(r.max_num_seqs),
        "--enable-lora",
        "--max-lora-rank",
        str(r.max_lora_rank),
    ]
    vp = Path(list(importlib.util.find_spec("vllm").submodule_search_locations)[0])
    parser = "qwen3" if list(vp.glob("**/qwen3_reasoning_parser.py")) else "deepseek_r1"
    args += ["--reasoning-parser", parser]
    if "--enable-reasoning" in help_text:
        args += ["--enable-reasoning"]
    if "--generation-config" in help_text:
        args += ["--generation-config", "vllm"]
    if r.enforce_eager:
        args += ["--enforce-eager"]
    if r.enable_prefix_caching:
        args += ["--enable-prefix-caching"]
    served = [base["id"]]
    if model["model_kind"] == "lora":
        ac = json.loads((Path(model["path"]) / "adapter_config.json").read_text())
        check(ac["r"] <= r.max_lora_rank, "adapter rank超过实例限制")
        args += ["--lora-modules", f"{model['id']}={model['path']}"]
        served.append(model["id"])
    for k, v in r.extra_args.items():
        flag = k.replace("_", "-")
        check(flag in VLLM_EXTRA and "--" + flag in help_text, f"不支持vLLM参数:{k}")
        if isinstance(v, bool):
            if v:
                args.append("--" + flag)
        else:
            check(isinstance(v, (str, int, float)), "vLLM参数需标量")
            args += ["--" + flag, str(v)]
    db.update(
        job["id"],
        internal_url=url,
        internal_key=key,
        served_models=served,
        base_model_id=base["id"],
        parser=parser,
    )
    with httpx.Client(timeout=2, trust_env=False) as client:

        def ready():
            try:
                if (
                    client.get(
                        url + "/health", headers={"Authorization": "Bearer " + key}
                    ).status_code
                    == 200
                ):
                    state(db, job["id"], "ready", ready_at=stamp())
                    return True
            except httpx.HTTPError:
                pass
            return False

        logged_process(
            db,
            job,
            args,
            env=subenv(
                CUDA_VISIBLE_DEVICES=",".join(job["gpu_uuids"]),
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
            ),
            ready=ready,
        )
    return {"stopped": True}


def worker():
    db = DB()
    job = db.get(os.environ["VERL_API_JOB"], "job")
    locks = []
    try:
        p = db.root / "locks"
        p.mkdir(exist_ok=True)
        for gpu in sorted(job.get("gpu_uuids", [])):
            f = (p / (gpu + ".lock")).open("a")
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(f)
        db.update(
            job["id"],
            pid=os.getpid(),
            pid_start=proc_start(os.getpid()),
            process_boot_id=process_boot_id(),
            started_at=stamp(),
        )
        state(db, job["id"], "loading" if job["job_type"] == "inference" else "running")
        kind = job["job_type"]
        if kind == "prepare":
            result = prepare_data(db, job)
        elif kind == "import":
            result = append(db, job["dataset_id"], rows(job["snapshot"]))
        elif kind == "export":
            result = export_model(db, job)
        elif kind == "inference":
            result = inference_worker(db, job)
        elif kind == "train":
            env = subenv(
                VERL_API_MODE="rank",
                VERL_API_JOB=job["id"],
                CUDA_VISIBLE_DEVICES=",".join(job["gpu_uuids"]),
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
            )
            env.update(job["request"].get("runtime_env", {}))
            logged_process(
                db,
                job,
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={len(job['gpu_uuids'])}",
                    str(SCRIPT),
                ],
                env=env,
                cwd=str(verl_path().parent),
            )
            result = {
                "checkpoint_ids": [
                    c["id"]
                    for c in db.all("checkpoint")
                    if c["job_id"] == job["id"] and c["state"] != "missing"
                ],
                "output_dir": job["output_dir"],
            }
        else:
            raise ValueError("未知任务类型")
        state(
            db,
            job["id"],
            "cancelled" if db.get(job["id"])["state"] == "stopping" else "succeeded",
            result=result,
            finished_at=stamp(),
        )
    except BaseException as e:
        traceback.print_exc()
        cancel = db.get(job["id"])["state"] == "stopping" or isinstance(
            e, (SystemExit, KeyboardInterrupt)
        )
        state(
            db,
            job["id"],
            "cancelled" if cancel else "failed",
            error={"code": getattr(e, "code", "WORKER_FAILED"), "message": str(e)},
            finished_at=stamp(),
        )
    finally:
        for f in locks:
            f.close()


# Storage v1: only server-discovered artifacts may be removed, never arbitrary paths.
def storage_process_live(job):
    # Unknown ownership is blocked explicitly, never reported as proven alive.
    return job_process_status(job)["status"] != "exited"


def storage_nested_mounts(path):
    # ismount() alone misses same-filesystem bind mounts on Linux.
    p = Path(path).absolute()
    found = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if mount == p or mount.is_relative_to(p):
                found.append(str(mount))
    except (OSError, IndexError):
        # Fail closed for Linux-managed destructive operations if mount inventory fails.
        return ["无法核验/proc/self/mountinfo"]
    return found


class CleanupPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_ids: list[str] = Field(default_factory=list, max_length=500)
    job_id: str | None = None
    mode: Literal["selected", "old_checkpoints", "job_outputs", "records"] = "selected"
    record_ids: list[str] = Field(default_factory=list, max_length=500)
    keep_last: int = Field(2, ge=0)


class CleanupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str
    confirm: Literal[True]


def storage_overlap(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def storage_measure(path, details=False):
    """Metadata-only scan, no symlink traversal; signature detects stale previews."""
    p = Path(path)
    result = dict(size_bytes=0, allocated_bytes=0, file_count=0,
                  modified_at=None, scan_errors=[], has_symlinks=False, has_mounts=False, files=[])
    digest = hashlib.sha256()
    seen = set()
    newest = 0

    def visit(f):
        nonlocal newest
        try:
            st = f.lstat()
            rel = str(f.relative_to(p)) if f != p else "."
            digest.update(js([rel, st.st_dev, st.st_ino, st.st_mode, st.st_size,
                              st.st_mtime_ns, st.st_ctime_ns, st.st_nlink]).encode())
            newest = max(newest, st.st_mtime)
            if f.is_dir() and os.path.ismount(f):
                result["has_mounts"] = True
            if stat.S_ISLNK(st.st_mode):
                result["has_symlinks"] = True
                return
            if not stat.S_ISREG(st.st_mode):
                return
            result["file_count"] += 1
            if (st.st_dev, st.st_ino) not in seen:
                result["size_bytes"] += st.st_size
                result["allocated_bytes"] += st.st_blocks * 512
                seen.add((st.st_dev, st.st_ino))
            if details:
                result["files"].append(dict(path=rel, size_bytes=st.st_size,
                    allocated_bytes=st.st_blocks * 512, hardlink_count=st.st_nlink))
        except OSError as e:
            result["scan_errors"].append(str(e))

    visit(p)
    if p.is_dir() and not p.is_symlink():
        def error(e):
            result["scan_errors"].append(str(e))
        for root, dirs, files in os.walk(p, followlinks=False, onerror=error):
            dirs.sort()
            for name in sorted(dirs + files):
                visit(Path(root) / name)
    result["signature"] = digest.hexdigest()
    result["modified_at"] = datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest else None
    if not details:
        result.pop("files")
    return result


class StorageLeaseMiddleware:
    def __init__(self, app, storage):
        self.app, self.storage = app, storage

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        guarded = scope["type"] == "http" and (
            any(path.startswith("/api/v1/" + prefix) for prefix in
                ("datasets", "dataset-versions", "models", "exports", "inference"))
            or (path.startswith("/api/v1/jobs") and (scope.get("method") == "POST" or path.endswith("/logs"))))
        if not guarded:
            return await self.app(scope, receive, send)
        with self.storage.lease_lock:
            busy = self.storage.cleaning
            if not busy:
                self.storage.readers += 1
        if busy:
            from starlette.responses import JSONResponse
            return await JSONResponse({"error": {"code": "STORAGE_BUSY", "message": "清理进行中，请稍后重试"}},
                                      status_code=409)(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        finally:
            with self.storage.lease_lock:
                self.storage.readers -= 1


class Storage:
    def __init__(self, manager):
        self.m, self.db = manager, manager.db
        self.lock = threading.RLock()
        self.scan_state = {"state": "not_scanned", "scanned_at": None, "items": []}
        self.cleaning = False
        self.lease_lock = threading.Lock()
        self.readers = 0
        self.threads = []

    def recover(self):
        # Never replay an interrupted destructive operation automatically.
        for kind in ("storage_plan", "storage_cleanup"):
            for obj in self.db.all(kind):
                if obj["state"] in ("running", "queued", "executing"):
                    self.db.update(obj["id"], state="interrupted",
                        error="服务重启打断清理；请重新扫描并预览。已删除部分不会恢复。")
        for kind in ("checkpoint", "model", "dataset_version", "dataset", "job"):
            for obj in self.db.all(kind):
                if obj.get("storage_state") == "deleting":
                    self.db.update(obj["id"], state="missing" if kind == "checkpoint" else "deleted",
                        storage_state="interrupted", storage_deleted_at=stamp(),
                        **({"resumable": False} if kind == "checkpoint" else {}))

    def available(self, obj):
        check(obj.get("storage_state") not in ("deleting", "deleted", "interrupted", "delete_failed")
              and obj.get("state") not in ("deleted", "missing"),
              "产物已删除、清理中或不完整: " + obj["id"], "ARTIFACT_UNAVAILABLE", 409)
        check(Path(obj["path"]).exists(), "产物文件不存在: " + obj["id"], "ARTIFACT_UNAVAILABLE", 409)

    def model_paths(self, id, visited=None):
        visited = set() if visited is None else visited
        if id in visited:
            return []
        visited.add(id)
        obj = self.db.get(id, "model")
        paths = [obj["path"], obj.get("tokenizer_path", obj["path"])]
        if obj.get("base_model_id"):
            paths += self.model_paths(obj["base_model_id"], visited)
        return paths

    def admit(self, request, extra=None):
        """Called under manager lock at enqueue, not merely at earlier validation."""
        check(not self.cleaning, "清理进行中，请完成后重试", "STORAGE_BUSY", 409)
        values = {**request, **(extra or {})}
        if values.get("dataset_id"):
            obj = self.db.get(values["dataset_id"], "dataset")
            check(obj.get("state") != "deleted", "数据集已删除", "ARTIFACT_UNAVAILABLE", 409)
        for field, kind in (("model_id", "model"), ("dataset_version_id", "dataset_version"),
                            ("checkpoint_id", "checkpoint"), ("resumed_from", "checkpoint")):
            if values.get(field):
                obj = self.db.get(values[field], kind)
                self.available(obj)
                if kind == "model":
                    for path in self.model_paths(obj["id"]):
                        check(Path(path).exists(), "模型依赖文件不存在", "ARTIFACT_UNAVAILABLE", 409)

    def catalog(self):
        items = {}
        def add(path, kind, obj=None, job=None, readonly=False):
            p = Path(path).absolute()
            if not p.exists() and not p.is_symlink():
                return
            key = "artifact_" + hashlib.sha256(str(p).encode()).hexdigest()[:24]
            if key in items:
                if obj:
                    items[key]["object_ids"].append(obj["id"])
                items[key]["readonly"] |= readonly
                return
            items[key] = dict(artifact_id=key, type=kind, path=str(p),
                job_id=job, object_ids=[obj["id"]] if obj else [], readonly=readonly,
                step=obj.get("step") if obj else None)
        models = self.db.all("model")
        for obj in models:
            add(obj["path"], "model", obj, obj.get("export_job_id"),
                readonly=obj["id"] == "base" or not obj.get("export_job_id"))
        for obj in self.db.all("checkpoint"):
            add(obj["path"], "checkpoint", obj, obj.get("job_id"))
        for obj in self.db.all("dataset_version"):
            add(obj["path"], "dataset_version", obj)
        for job in self.db.all("job"):
            out = Path(job["output_dir"])
            if out.is_dir() and not out.is_symlink():
                for child in sorted(out.iterdir()):
                    # Do not create a parent residual encompassing a known nested artifact.
                    if not any(storage_overlap(child, a["path"]) for a in items.values()):
                        add(child, "residual", job=job["id"])
            if job.get("version_id"):
                add(self.db.root / "dataset_versions" / job["version_id"],
                    "dataset_version", job=job["id"])
                for a in items.values():
                    if a["path"] == str(self.db.root / "dataset_versions" / job["version_id"]):
                        a["job_id"] = job["id"]
            if job.get("snapshot"):
                add(job["snapshot"], "snapshot", job=job["id"])
            if job.get("log_path"):
                add(job["log_path"], "log", job=job["id"], readonly=True)
        for obj in self.db.all("dataset"):
            add(self.db.root / "datasets" / obj["id"], "raw_dataset", obj, readonly=True)
        return items

    def blockers(self, a, deleting_records=()):
        p = Path(a["path"])
        reasons, active = [], []
        root = self.db.root.resolve()
        allowed = any(p.resolve().is_relative_to(root / name)
                      for name in (("jobs", "dataset_versions", "snapshots", "uploads", "datasets")
                                   if a.get("record_delete") else ("jobs", "dataset_versions", "snapshots", "uploads")))
        if a["readonly"] or not allowed:
            reasons.append("基座/外部注册模型、原始数据和日志只读；仅清理API管理的产物")
        mounts = storage_nested_mounts(p)
        if mounts:
            reasons.append("目标包含挂载点或无法核验挂载: " + ", ".join(mounts))
        # Reject symlink ancestors and targets, including links within the data root.
        if any(q.is_symlink() for q in [p, *p.parents] if q != root.parent):
            reasons.append("路径包含符号链接，禁止删除")
        for obj in self.db.all("model"):
            if obj.get("state") == "deleted" or obj["id"] in deleting_records:
                continue
            if obj["id"] == "base" or not obj.get("export_job_id"):
                if any(storage_overlap(p, q) for q in self.model_paths(obj["id"])):
                    reasons.append("与受保护模型或其tokenizer/底座重叠: " + obj["id"])
            elif obj["id"] not in a["object_ids"]:
                deps = [obj.get("tokenizer_path", obj["path"])]
                if obj.get("base_model_id"):
                    deps += self.model_paths(obj["base_model_id"])
                if any(storage_overlap(p, q) for q in deps):
                    reasons.append("仍被模型依赖: " + obj["id"])
        for obj in self.db.all("checkpoint"):
            if obj["id"] not in deleting_records and obj.get("state") not in ("missing", "deleted") and Path(obj["path"]).exists():
                for key in ("base_model_id", "model_id"):
                    if obj.get(key) and any(storage_overlap(p, q) for q in self.model_paths(obj[key])):
                        reasons.append("检查点恢复/导出仍依赖此模型: " + obj["id"])
        for obj in self.db.all("dataset_version"):
            if obj.get("state") != "deleted" and obj["id"] not in a["object_ids"] and obj["id"] not in deleting_records:
                if obj.get("tokenizer_path") and storage_overlap(p, obj["tokenizer_path"]):
                    reasons.append("仍被数据预览的tokenizer引用: " + obj["id"])
        for job in self.db.all("job"):
            if job["state"] in DONE | {"deleted"} and not storage_process_live(job):
                continue
            paths = [job["output_dir"]]
            if job.get("dataset_id"):
                paths.append(str(root / "datasets" / job["dataset_id"]))
            if job.get("snapshot"):
                paths.append(job["snapshot"])
            if job.get("version_id"):
                paths.append(str(root / "dataset_versions" / job["version_id"]))
            req = job.get("request", {})
            ids = set(job.get("served_models", []))
            if req.get("model_id"):
                ids.add(req["model_id"])
            for mid in ids:
                paths += self.model_paths(mid)
            for ref in (req.get("checkpoint_id"), req.get("dataset_version_id"), job.get("resumed_from")):
                if ref:
                    obj = self.db.get(ref)
                    paths.append(obj["path"])
                    if obj["kind"] == "checkpoint":
                        for key in ("base_model_id", "model_id"):
                            if obj.get(key):
                                paths += self.model_paths(obj[key])
            if any(storage_overlap(p, q) for q in paths):
                active.append(job["id"])
        if active:
            reasons.append("任务正在使用或排队引用该产物")
        if any(self.m.chat_counts.values()):
            reasons.append("推理请求/adapter加载进行中，请稍后重试")
        with self.lease_lock:
            if self.readers:
                reasons.append("文件读取或对象操作请求进行中，请稍后重试")
        return reasons, active

    def inspect(self, a, details=False, deleting_records=()):
        result = {**a, **storage_measure(a["path"], details)}
        reasons, active = self.blockers(a, deleting_records)
        if result["has_symlinks"]:
            reasons.append("目录内包含符号链接，第一版禁止删除")
        if result["has_mounts"]:
            reasons.append("产物包含挂载点，禁止递归删除")
        if result["scan_errors"]:
            reasons.append("扫描不完整，请先处理扫描错误")
        result.update(deletable=not reasons, blocked_reasons=reasons, active_job_ids=active,
                      process_blockers=[{"job_id": jid, **job_process_status(self.db.get(jid, "job"))} for jid in active])
        return result

    def scan(self):
        with self.lock:
            if self.scan_state["state"] == "running":
                return self.summary()
            self.scan_state.update(state="running", started_at=stamp(), error=None)
        def run():
            try:
                items = [self.inspect(a) for a in self.catalog().values()]
                with self.lock:
                    self.scan_state = dict(state="succeeded", scanned_at=stamp(), items=items, error=None)
            except Exception as e:
                with self.lock:
                    self.scan_state.update(state="failed", error=str(e))
        t = threading.Thread(target=run, daemon=True)
        self.threads.append(t)
        t.start()
        return self.summary()

    def summary(self):
        with self.lock:
            snap = dict(self.scan_state)
        items = snap.pop("items")
        categories = {}
        for a in items:
            c = categories.setdefault(a["type"], dict(size_bytes=0, allocated_bytes=0, count=0))
            for k in ("size_bytes", "allocated_bytes"):
                c[k] += a[k]
            c["count"] += 1
        volumes = {}
        for path in [str(self.db.root), *[a["path"] for a in items]]:
            try:
                device = str(Path(path).stat().st_dev)
                if device not in volumes:
                    d = shutil.disk_usage(path)
                    volumes[device] = dict(device=device, path=path, total_bytes=d.total,
                        used_bytes=d.used, free_bytes=d.free)
            except OSError:
                pass
        return {**snap, "volumes": list(volumes.values()), "categories": categories,
                "artifact_count": len(items), "cleanup_running": self.cleaning,
                "note": "分类统计仅覆盖已发现产物；不同产物间硬链接可能重复计数；实际释放量以磁盘变化为准。"}

    def record_selection(self, ids):
        check(ids, "请选择record_ids", status=422)
        selected = {i: self.db.get(i) for i in dict.fromkeys(ids)}
        allowed = {"dataset", "dataset_version", "job", "checkpoint", "model"}
        check(all(o["kind"] in allowed for o in selected.values()), "不支持此对象类型", status=422)
        # A task owns its checkpoints/export or prepared version; show expansion explicitly.
        for obj in list(selected.values()):
            if obj["kind"] == "job":
                for kind, key in (("checkpoint", "job_id"), ("model", "export_job_id")):
                    for child in self.db.all(kind):
                        if child.get(key) == obj["id"]:
                            selected[child["id"]] = child
                if obj.get("version_id"):
                    for dv in self.db.all("dataset_version"):
                        if dv["id"] == obj["version_id"]:
                            selected[dv["id"]] = dv
        return selected

    def record_blockers(self, selected):
        reasons = []
        ids = set(selected)
        for oid, obj in selected.items():
            if oid == "base" or (obj["kind"] == "model" and not obj.get("export_job_id")):
                reasons.append({"id": oid, "reason": "基座和外部注册模型受保护"})
            if obj["kind"] == "job":
                report = job_process_status(obj)
                if obj["state"] not in DONE | {"deleted"} or report["status"] != "exited":
                    reasons.append({"id": oid, "reason": "进程归属无法核验" if report["status"] == "unknown"
                                    else "任务未结束或有已核实的存活进程", "process_status": report})
            if obj["kind"] == "dataset":
                for dv in self.db.all("dataset_version"):
                    if dv.get("dataset_id") == oid and dv["id"] not in ids:
                        reasons.append({"id": oid, "reason": "仍有数据版本，请一并选择或先删除", "dependent_id": dv["id"]})
        # Check IDs even when an artifact is already missing on disk.
        for j in self.db.all("job"):
            report = job_process_status(j)
            if j["state"] in DONE | {"deleted"} and report["status"] == "exited":
                continue
            req = j.get("request", {})
            refs = set(j.get("served_models", []))
            refs.update(v for v in (j.get("dataset_id"), j.get("version_id"), j.get("resumed_from"),
                        req.get("model_id"), req.get("dataset_version_id"), req.get("checkpoint_id")) if v)
            for oid in ids & refs:
                reasons.append({"id": oid, "reason": "活跃或排队任务仍引用此记录" if report["status"] != "unknown" else "引用任务的进程归属无法核验", "dependent_id": j["id"], "process_status": report})
        for model in self.db.all("model"):
            if model["id"] not in ids and model.get("state") != "deleted" and model.get("base_model_id") in ids:
                reasons.append({"id": model["base_model_id"], "reason": "其他模型仍依赖此底座", "dependent_id": model["id"]})
        for ck in self.db.all("checkpoint"):
            if ck["id"] not in ids and ck["state"] not in ("missing", "deleted"):
                for key in ("base_model_id", "model_id"):
                    if ck.get(key) in ids:
                        reasons.append({"id": ck[key], "reason": "检查点恢复/导出仍依赖此模型", "dependent_id": ck["id"]})
        if any(self.m.chat_counts.values()):
            reasons.append({"id": None, "reason": "推理请求或adapter操作尚未结束"})
        with self.lease_lock:
            if self.readers:
                reasons.append({"id": None, "reason": "文件读取或对象操作尚未结束"})
        return reasons

    def record_items(self, selected):
        paths = []
        root = self.db.root.resolve()
        for obj in selected.values():
            kind = obj["kind"]
            if kind == "dataset":
                p = root / "datasets" / obj["id"]
            elif kind == "job":
                p = root / "jobs" / obj["id"]
            else:
                p = Path(obj["path"])
            if p.exists() or p.is_symlink():
                paths.append((p, obj))
            if kind == "job":
                # Source copies and failed prepare outputs are also owned by the task.
                extras = [obj.get("snapshot")]
                if obj.get("version_id"):
                    extras.append(str(root / "dataset_versions" / obj["version_id"]))
                for extra in extras:
                    if extra and (Path(extra).exists() or Path(extra).is_symlink()):
                        paths.append((Path(extra), obj))
        # Collapse a task directory and its child checkpoints into one deletion target.
        roots = []
        for p, obj in sorted(paths, key=lambda pair: len(pair[0].parts)):
            parent = next((a for a in roots if p == Path(a["path"]) or p.is_relative_to(Path(a["path"]))), None)
            if parent:
                if obj["id"] not in parent["object_ids"]:
                    parent["object_ids"].append(obj["id"])
            else:
                roots.append(dict(artifact_id="artifact_" + hashlib.sha256(str(p).encode()).hexdigest()[:24],
                    type="record_files", path=str(p), object_ids=[obj["id"]],
                    job_id=obj["id"] if obj["kind"] == "job" else obj.get("job_id"),
                    step=obj.get("step"), readonly=False, record_delete=True))
        return roots

    def records_preview(self, request):
        with self.m.lock:
            check(not self.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            selected = self.record_selection(request.record_ids)
            blocked = self.record_blockers(selected)
            items = [self.inspect(a, deleting_records=set(selected)) for a in self.record_items(selected)]
            records = [dict(id=o["id"], kind=o["kind"], name=o.get("name", o.get("request", {}).get("name")),
                job_type=o.get("job_type"), state=o["state"],
                signature=hashlib.sha256(js(o).encode()).hexdigest()) for o in selected.values()]
            return self.db.add("storage_plan", dict(mode="records", requested_record_ids=request.record_ids,
                records=records, blocked_records=blocked, items=items, expires_at=time.time()+900,
                executable=not blocked and all(a["deletable"] for a in items),
                estimated_bytes=sum(a["allocated_bytes"] for a in items if a["deletable"]),
                effects=["永久删除所列记录及文件，不进入回收站",
                         "删除任务/推理历史时，同时删除其日志、loss和事件；普通文件清理仍保留这些历史",
                         "任务所属检查点、导出模型或准备的数据版本会明确列入本计划",
                         "不会连带删除其他独立任务；依赖阻止项须先处理"]), state="ready")

    def recheck_records(self, plan):
        selected = self.record_selection(plan["requested_record_ids"])
        check(set(selected) == {r["id"] for r in plan["records"]}, "关联记录变化，请重新预览", "PLAN_STALE", 409)
        check(not self.record_blockers(selected), "记录被使用或存在依赖，请重新预览", "ARTIFACT_IN_USE", 409)
        for r in plan["records"]:
            check(hashlib.sha256(js(selected[r["id"]]).encode()).hexdigest() == r["signature"],
                  "记录发生变化，请重新预览", "PLAN_STALE", 409)
        current = {a["artifact_id"]: a for a in self.record_items(selected)}
        check(set(current) == {a["artifact_id"] for a in plan["items"]}, "文件清单变化，请重新预览", "PLAN_STALE", 409)
        return current

    def remove_records(self, plan):
        # Only commit metadata deletion when every planned file deletion succeeded.
        with self.db.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for record in plan["records"]:
                oid = record["id"]
                if record["kind"] == "job":
                    c.execute("DELETE FROM metrics WHERE ref=?", (oid,))
                    c.execute("DELETE FROM events WHERE ref=?", (oid,))
                c.execute("DELETE FROM objects WHERE id=?", (oid,))
        self.db.emit(plan["id"], "storage_records_deleted", {"record_ids": [r["id"] for r in plan["records"]]})


    def preview(self, request):
        if request.mode == "records":
            return self.records_preview(request)
        with self.m.lock:
            check(not self.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            catalog = self.catalog()
            if request.mode == "selected":
                check(request.artifact_ids, "请选择artifact_ids", status=422)
                check(all(i in catalog for i in request.artifact_ids), "产物不存在，请重新扫描", "NOT_FOUND", 404)
                chosen = [catalog[i] for i in dict.fromkeys(request.artifact_ids)]
            else:
                check(request.job_id, "此模式需要job_id", status=422)
                self.db.get(request.job_id, "job")
                chosen = [a for a in catalog.values() if a["job_id"] == request.job_id
                          and a["type"] in ("checkpoint", "model", "residual")]
                if request.mode == "old_checkpoints":
                    chosen = sorted([a for a in chosen if a["type"] == "checkpoint"],
                        key=lambda a: a["step"] or 0, reverse=True)[request.keep_last:]
            check(chosen, "没有匹配的清理产物", "EMPTY_SELECTION", 422)
            for i, a in enumerate(chosen):
                check(not any(storage_overlap(a["path"], b["path"]) for b in chosen[:i]),
                      "清理目标重叠，请分别选择", "OVERLAPPING_ARTIFACTS", 409)
            items = [self.inspect(a) for a in chosen]
            return self.db.add("storage_plan", dict(items=items, expires_at=time.time() + 900,
                executable=all(a["deletable"] for a in items),
                estimated_bytes=sum(a["allocated_bytes"] for a in items if a["deletable"]),
                effects=["永久删除文件，不进入回收站", "保留任务记录、日志、loss和API Key",
                         "删除检查点后不能恢复/重新导出；已独立导出的模型保留",
                         "删除模型或数据版本后，相应加载/训练/预览不可用"]), state="ready")

    def execute(self, request):
        with self.m.lock:
            check(not self.cleaning, "已有清理进行中", "STORAGE_BUSY", 409)
            plan = self.db.get(request.plan_id, "storage_plan")
            check(plan["state"] == "ready" and plan["expires_at"] > time.time(),
                  "计划已使用或过期，请重新预览", "PLAN_EXPIRED", 409)
            check(plan["executable"], "预览包含禁止删除项，请重新选择", "ARTIFACT_IN_USE", 409)
            deleting_records = {r["id"] for r in plan.get("records", [])}
            catalog = self.recheck_records(plan) if plan.get("mode") == "records" else self.catalog()
            for old in plan["items"]:
                check(old["artifact_id"] in catalog, "产物发生变化，请重新预览", "PLAN_STALE", 409)
                current = self.inspect(catalog[old["artifact_id"]], deleting_records=deleting_records)
                check(current["deletable"], js(current["blocked_reasons"]), "ARTIFACT_IN_USE", 409)
                check(current["signature"] == old["signature"] and current["object_ids"] == old["object_ids"],
                      "文件或引用已变化，请重新预览", "PLAN_STALE", 409)
            with self.lease_lock:
                check(not self.readers, "文件读取或对象操作进行中", "STORAGE_BUSY", 409)
                self.cleaning = True
            try:
                result = self.db.add("storage_cleanup", dict(plan_id=plan["id"], items=[],
                    completed=0, total=len(plan["items"]), deleted_bytes=0), state="queued")
                self.db.update(plan["id"], state="executing", cleanup_id=result["id"])
                t = threading.Thread(target=self._delete, args=(result["id"], plan), daemon=True)
                self.threads.append(t)
                t.start()
                return result
            except BaseException:
                with self.lease_lock:
                    self.cleaning = False
                raise


    def _delete(self, id, plan):
        results = []
        before = shutil.disk_usage(self.db.root).free
        try:
            self.db.update(id, state="running")
            for a in plan["items"]:
                item = dict(artifact_id=a["artifact_id"], path=a["path"], state="failed", deleted_bytes=0)
                touched = False
                try:
                    with self.m.lock:
                        current = self.inspect(a, deleting_records={r["id"] for r in plan.get("records", [])})
                        check(current["deletable"] and current["signature"] == a["signature"],
                              "产物变化或被占用，停止删除该项", "PLAN_STALE", 409)
                        p = Path(a["path"])
                        # Tombstone before unlink; interruption must never leave a usable record.
                        for oid in a["object_ids"]:
                            obj = self.db.get(oid)
                            self.db.update(oid, storage_state="deleting",
                                state="missing" if obj["kind"] == "checkpoint" else "deleted",
                                **({"resumable": False} if obj["kind"] == "checkpoint" else {}))
                        touched = True
                        relative = p.relative_to(self.db.root.resolve())
                        fd = os.open(self.db.root.resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                        try:
                            for component in relative.parts[:-1]:
                                child_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                                os.close(fd)
                                fd = child_fd
                            name = relative.parts[-1]
                            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                            check(not stat.S_ISLNK(info.st_mode), "目标变成符号链接", "PLAN_STALE", 409)
                            if stat.S_ISDIR(info.st_mode):
                                check(shutil.rmtree.avoids_symlink_attacks,
                                      "此平台不支持安全递归删除", "UNSUPPORTED_PLATFORM", 503)
                                shutil.rmtree(name, dir_fd=fd)
                            else:
                                os.unlink(name, dir_fd=fd)
                        finally:
                            os.close(fd)
                        for oid in a["object_ids"]:
                            self.db.update(oid, storage_state="deleted", storage_deleted_at=stamp())
                        item.update(state="succeeded", deleted_bytes=a["allocated_bytes"])
                except Exception as e:
                    item["error"] = str(e)
                    if touched:
                        for oid in a["object_ids"]:
                            self.db.update(oid, storage_state="delete_failed")
                results.append(item)
                self.db.update(id, items=results, completed=len(results),
                    deleted_bytes=sum(r["deleted_bytes"] for r in results))
            status = "succeeded" if all(r["state"] == "succeeded" for r in results) else "failed"
            if status == "succeeded" and plan.get("mode") == "records":
                with self.m.lock:
                    self.remove_records(plan)
            self.db.update(id, state=status, finished_at=stamp(),
                free_bytes_delta=shutil.disk_usage(self.db.root).free - before,
                note="deleted_bytes为完整删除项的分配大小；free_bytes_delta含其他并发磁盘活动。")
            self.db.update(plan["id"], state=status)
        except Exception as e:
            self.db.update(id, state="failed", error=str(e), items=results)
            self.db.update(plan["id"], state="failed")
        finally:
            with self.lease_lock:
                self.cleaning = False
            self.scan()



class Manager:
    def __init__(self, model_path, root=ROOT):
        self.db = DB(root)
        self.model_path = str(Path(model_path).resolve())
        self.lock = threading.RLock()
        self.storage = Storage(self)
        self.gpu_lock = threading.Lock()
        self.probe_lock = threading.Lock()
        self.children = {}
        self.chat_tasks = {}
        self.chat_counts = {}
        self.closed = threading.Event()
        self.thread = None
        self.filelock = None
        self.environment = None
        self.gpu_time = 0
        self.gpu_value = {"available": False, "gpus": [], "error": "尚未采样"}
        key = self.db.root / "api_key"
        if not key.exists():
            try:
                fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(secrets.token_urlsafe(32) + "\n")
            except FileExistsError:
                pass
        self.key = key.read_text().strip()
        try:
            base = self.db.get("base", "model")
            check(
                base["path"] == self.model_path,
                "此数据目录base已绑定其他路径；请用models/register添加模型",
            )
        except Error as e:
            if e.code != "NOT_FOUND":
                raise
            self.db.add(
                "model",
                {
                    "name": Path(model_path).name,
                    "model_kind": "full",
                    "path": self.model_path,
                    "tokenizer_path": self.model_path,
                    "base_model_id": None,
                    "chat_defaults": {},
                },
                id="base",
            )

    def start(self):
        self.filelock = (self.db.root / "server.lock").open("a")
        try:
            fcntl.flock(self.filelock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有API使用此数据目录，不能启多个worker")
        atomic(
            self.db.root / "server_process.json",
            {
                "pid": os.getpid(),
                "pid_start": proc_start(os.getpid()),
                "model_path": self.model_path,
                "script": str(SCRIPT),
                "started_at": stamp(),
            },
        )
        self.storage.recover()
        self.thread = threading.Thread(target=self.schedule, daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=10)
        for thread in self.storage.threads:
            thread.join()
        if self.filelock:
            self.filelock.close()

    def probe(self):
        with self.probe_lock:
            if self.environment is None:
                p = self.db.root / "probe.json"
                command(
                    [sys.executable, str(SCRIPT)],
                    timeout=150,
                    env=subenv(VERL_API_MODE="probe", VERL_API_OUTPUT=str(p)),
                )
                self.environment = json.loads(p.read_text())
            return self.environment

    def gpus(self, force=False):
        with self.gpu_lock:
            if (
                force
                or time.monotonic() - self.gpu_time
                >= settings(self.db)["gpu_poll_seconds"]
            ):
                self.gpu_value = gpu_snapshot(self.db)
                self.gpu_time = time.monotonic()
            return self.gpu_value

    def job(self, kind, request, **kw):
        with self.lock:
            self.storage.admit(request, kw)
            return self._job(kind, request, **kw)

    def _job(self, kind, request, **kw):
        id = ident("job")
        p = self.db.root / "jobs" / id
        p.mkdir(parents=True)
        obj = self.db.add(
            "job",
            {
                "job_type": kind,
                "request": request,
                "result": None,
                "error": None,
                "progress": {},
                "gpu_uuids": [],
                "wait_reason": None,
                "log_path": str(p / "job.log"),
                "output_dir": str(p / "output"),
                **kw,
            },
            state="queued",
            id=id,
        )
        self.db.emit(id, "state", {"state": "queued"})
        return obj

    def validate(self, r, resume=None):
        with self.lock:
            self.storage.admit(r.model_dump())
        cfg = training_config(
            self.db,
            r.model_dump(),
            self.db.root / "jobs" / "<job_id>" / "output",
            self.probe(),
            resume,
        )
        p = self.db.root / "validation" / (ident("v") + ".json")
        q = p.with_suffix(".result.json")
        atomic(p, cfg["overrides"])
        try:
            command(
                [sys.executable, str(SCRIPT)],
                timeout=60,
                env=subenv(
                    VERL_API_MODE="validate",
                    VERL_API_INPUT=str(p),
                    VERL_API_OUTPUT=str(q),
                ),
            )
            cfg["resolved_config"] = json.loads(q.read_text())
        except Error as e:
            raise Error(e.message, "INVALID_VERL_CONFIG", 400)
        finally:
            p.unlink(missing_ok=True)
            q.unlink(missing_ok=True)
        cfg["gpu_uuids"] = select_gpus(r.gpu_ids, self.gpus(True))
        cfg["argv"] = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(r.gpu_ids)}",
            str(SCRIPT),
        ]
        cfg["launch_note"] = (
            "内部rank入口读取resolved_config后调用verl.trainer.sft_trainer.run_sft"
        )
        return cfg

    def train(self, r, cfg, **extra):
        with self.lock:
            job = self.job("train", r.model_dump(), gpu_uuids=cfg["gpu_uuids"], **extra)
            cfg["overrides"]["trainer.default_local_dir"] = job["output_dir"]
            cfg["resolved_config"]["trainer"]["default_local_dir"] = job["output_dir"]
            atomic(
                Path(job["output_dir"]).parent / "resolved_config.json",
                cfg["resolved_config"],
            )
            return self.db.update(job["id"], training_config=cfg)

    def cancel(self, id):
        with self.lock:
            j = self.db.get(id, "job")
            report = job_process_status(j)
            if report["status"] == "exited":
                if j["state"] in DONE:
                    return self.db.update(id, process_status=report)
                return state(self.db, id, "cancelled", finished_at=stamp(), process_status=report)
            if report["status"] == "unknown":
                return self.db.update(id, process_status=report,
                    wait_reason="进程归属无法核验；未向未知进程发信号，请查看process_status")
            # Even a terminal record must stop verified remaining children.
            j = state(self.db, id, "stopping", stop_requested_at=time.time())
            report = stop_process(j)
            return self.db.update(id, process_status=report)

    def schedule(self):
        while not self.closed.wait(0.5):
            try:
                pending = self.db.all("job")
                snap = (
                    self.gpus()
                    if any(
                        j["state"] == "queued" and j.get("gpu_uuids") for j in pending
                    )
                    else None
                )
                with self.lock:
                    for j in self.db.all("job"):
                        child = self.children.get(j["id"])
                        if child and child.poll() is not None:
                            self.children.pop(j["id"], None)
                        if j["state"] in LIVE:
                            process_report = job_process_status(j)
                            if process_report["status"] == "exited":
                                state(
                                    self.db,
                                    j["id"],
                                    "cancelled"
                                    if j["state"] == "stopping"
                                    else "interrupted",
                                    finished_at=stamp(),
                                    process_status=process_report,
                                    error={
                                        "code": "PROCESS_LOST",
                                        "message": "进程已退出，详见logs",
                                    },
                                )
                            elif process_report["status"] == "unknown":
                                self.db.update(j["id"], process_status=process_report,
                                    wait_reason="进程归属无法核验，保留任务及文件保护")
                            elif (
                                j["state"] == "stopping"
                                and time.time()
                                - j.get("stop_requested_at", time.time())
                                > 20
                            ):
                                self.db.update(j["id"], process_status=stop_process(j, signal.SIGKILL))
                    jobs = self.db.all("job")
                    active = [j for j in jobs if j["state"] in LIVE]
                    cpu = sum(not j.get("gpu_uuids") for j in active)
                    for j in reversed(jobs):
                        if j["state"] != "queued":
                            continue
                        if (
                            not j.get("gpu_uuids")
                            and cpu >= settings(self.db)["max_cpu_jobs"]
                        ):
                            continue
                        conflict = [
                            a
                            for a in active
                            if set(a.get("gpu_uuids", [])) & set(j.get("gpu_uuids", []))
                        ]
                        if conflict:
                            for a in conflict:
                                if (
                                    j["job_type"] == "train"
                                    and a["job_type"] == "inference"
                                    and a["state"] == "ready"
                                    and settings(self.db)["auto_unload_inference"]
                                    and not self.chat_counts.get(a["id"], 0)
                                ):
                                    self.cancel(a["id"])
                            self.db.update(
                                j["id"],
                                wait_reason="等待GPU任务释放:"
                                + ",".join(a["id"] for a in conflict),
                            )
                            continue
                        if j.get("gpu_uuids"):
                            if not snap or not snap["available"]:
                                self.db.update(j["id"], wait_reason="等待GPU可用")
                                continue
                            if any(
                                g["uuid"] in j["gpu_uuids"] and g["processes"]
                                for g in snap["gpus"]
                            ):
                                self.db.update(
                                    j["id"], wait_reason="GPU仍有外部或尚未释放的进程"
                                )
                                continue
                        state(self.db, j["id"], "starting", wait_reason=None)
                        with Path(j["log_path"]).open("a", encoding="utf-8") as out:
                            p = subprocess.Popen(
                                [sys.executable, str(SCRIPT)],
                                env=subenv(
                                    VERL_API_MODE="worker", VERL_API_JOB=j["id"]
                                ),
                                stdout=out,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True,
                            )
                        self.children[j["id"]] = p
                        updated = self.db.update(
                            j["id"], pid=p.pid, pid_start=proc_start(p.pid), process_boot_id=process_boot_id()
                        )
                        active.append(updated)
                        if not j.get("gpu_uuids"):
                            cpu += 1
            except Exception:
                traceback.print_exc()


def chat_params(r, defaults, budget):
    explicit = r.model_dump(exclude_unset=True)
    thinking = explicit.get("enable_thinking")
    if thinking is None:
        thinking = defaults.get("enable_thinking")
    if thinking is None:
        thinking = True
    p = {
        "enable_thinking": thinking,
        "temperature": 0.6 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 4096 if thinking else 2048,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": None,
        "stop": None,
        "thinking_token_budget": None,
    }
    for source in (defaults, explicit):
        for k, v in source.items():
            if k in ChatOptions.model_fields and (
                v is not None or k in ("seed", "stop", "thinking_token_budget")
            ):
                p[k] = v
    check(
        p["top_k"] == -1 or p["top_k"] >= 1,
        "top_k需-1或正整数",
        "INVALID_PARAMETER",
        422,
    )
    if p["thinking_token_budget"] is not None:
        check(
            budget, "当前vLLM不支持thinking_token_budget", "UNSUPPORTED_PARAMETER", 422
        )
        check(
            p["enable_thinking"] and p["thinking_token_budget"] < p["max_tokens"],
            "思考预算需要开启thinking，且小于总输出预算",
            "INVALID_PARAMETER",
            422,
        )
    return p


def reasoning(obj):
    for choice in obj.get("choices", []):
        for key in ("delta", "message"):
            value = choice.get(key)
            if isinstance(value, dict) and "reasoning" in value:
                value["reasoning_content"] = value.pop("reasoning")
    return obj


def safe_file(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    check(
        p.is_relative_to(root) and p.is_file(), "文件不存在或路径越界", "NOT_FOUND", 404
    )
    return p


def create_app(model_path=MODEL_DEFAULT, manager=None):
    import httpx
    from fastapi import APIRouter, Depends, FastAPI, File, Query, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import (
        FileResponse,
        JSONResponse,
        PlainTextResponse,
        StreamingResponse,
    )
    from fastapi.security import HTTPBearer

    globals().update(Request=Request, UploadFile=UploadFile)
    m = manager or Manager(model_path)
    db = m.db

    @contextlib.asynccontextmanager
    async def lifespan(app):
        m.start()
        try:
            yield
        finally:
            m.close()

    app = FastAPI(
        title="Verl CPT/SFT & Chat API",
        version=VERSION,
        lifespan=lifespan,
        description="完整接口规范/api-spec；请求字段、默认值、范围见本OpenAPI。",
    )
    app.state.manager = m
    app.add_middleware(StorageLeaseMiddleware, storage=m.storage)
    bearer = HTTPBearer(auto_error=False)

    def auth(credentials=Depends(bearer)):
        check(
            credentials
            and credentials.scheme.lower() == "bearer"
            and secrets.compare_digest(credentials.credentials, m.key),
            "需要Authorization: Bearer <api_key>",
            "UNAUTHORIZED",
            401,
        )

    api = APIRouter(dependencies=[Depends(auth)])

    @app.exception_handler(Error)
    async def error_handler(request, e):
        return JSONResponse(
            {
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": clean(e.details),
                }
            },
            status_code=e.status,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, e):
        return JSONResponse(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "请求字段不符合规范",
                    "details": clean(e.errors()),
                }
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def exception_handler(request, e):
        traceback.print_exc()
        return JSONResponse(
            {"error": {"code": "INTERNAL_ERROR", "message": str(e), "details": None}},
            status_code=500,
        )

    @app.middleware("http")
    async def cors(request, call_next):
        origin = request.headers.get("origin")
        allowed = settings(db)["cors_origins"]
        if origin and "*" not in allowed and origin not in allowed:
            return JSONResponse(
                {
                    "error": {
                        "code": "CORS_DENIED",
                        "message": "Origin未允许",
                        "details": None,
                    }
                },
                status_code=403,
            )
        response = (
            JSONResponse({"ok": True})
            if request.method == "OPTIONS"
            else await call_next(request)
        )
        if origin:
            response.headers.update(
                {
                    "Access-Control-Allow-Origin": origin,
                    "Vary": "Origin",
                    "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization,Content-Type,Last-Event-ID",
                    "Access-Control-Expose-Headers": "X-Request-ID",
                }
            )
        return response

    @app.get("/healthz", tags=["公开文档"])
    def health():
        return {
            "ok": True,
            "service": "verl-api",
            "version": VERSION,
            "timestamp": stamp(),
        }

    @app.get("/api-spec", response_class=PlainTextResponse, tags=["公开文档"])
    def specification():
        schema = app.openapi()
        routes = "\n".join(
            f"{method.upper()} {path}"
            for path, ops in schema["paths"].items()
            for method in ops
            if method in {"get", "post", "put", "delete", "patch"}
        )
        return (
            SPEC
            + "\n\n## 路由表\n```text\n"
            + routes
            + "\n```\n\n## 所有字段Schema（由同文件模型生成）\n```json\n"
            + json.dumps(
                schema.get("components", {}).get("schemas", {}),
                ensure_ascii=False,
                indent=2,
            )
            + "\n```\n"
        )

    @api.get("/api/v1/storage/records", tags=["存储管理"])
    def storage_records(
        type: Literal["dataset", "dataset_version", "train_job", "checkpoint", "model", "inference", "other_job"] | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
    ):
        items = []
        for kind in ("dataset", "dataset_version", "job", "checkpoint", "model"):
            for obj in db.all(kind):
                category = ({"train": "train_job", "inference": "inference"}.get(obj.get("job_type"), "other_job")
                            if kind == "job" else kind)
                if type and category != type:
                    continue
                protected = obj["id"] == "base" or (kind == "model" and not obj.get("export_job_id"))
                items.append(dict(id=obj["id"], type=category, state=obj["state"],
                    name=obj.get("name", obj.get("request", {}).get("name")),
                    created_at=obj["created_at"], protected=protected,
                    job_id=obj.get("job_id", obj.get("export_job_id")),
                    dataset_id=obj.get("dataset_id"), storage_state=obj.get("storage_state")))
        items.sort(key=lambda o: o["created_at"], reverse=True)
        return page(items, offset, limit)

    @api.get("/api/v1/storage/summary", tags=["存储管理"])
    def storage_summary():
        return m.storage.summary()

    @api.post("/api/v1/storage/scan", status_code=202, tags=["存储管理"])
    def storage_scan():
        return m.storage.scan()

    @api.get("/api/v1/storage/artifacts", tags=["存储管理"])
    def storage_artifacts(
        type: str | None = None, job_id: str | None = None,
        min_bytes: int = Query(0, ge=0),
        sort: Literal["size_desc", "modified_desc"] = "size_desc",
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
    ):
        with m.storage.lock:
            items = list(m.storage.scan_state["items"])
            scanned_at = m.storage.scan_state["scanned_at"]
        items = [a for a in items if (not type or a["type"] == type)
                 and (not job_id or a["job_id"] == job_id) and a["size_bytes"] >= min_bytes]
        items.sort(key=lambda a: a["size_bytes"] if sort == "size_desc"
                   else (a["modified_at"] or ""), reverse=True)
        return {**page(items, offset, limit), "scanned_at": scanned_at,
                "note": "缓存统计；deletable仅供展示，执行时会重新检查"}

    @api.get("/api/v1/storage/artifacts/{id}/files", tags=["存储管理"])
    def storage_files(id: str, offset: int = Query(0, ge=0),
                      limit: int = Query(100, ge=1, le=1000)):
        catalog = m.storage.catalog()
        check(id in catalog, "产物不存在", "NOT_FOUND", 404)
        item = m.storage.inspect(catalog[id], details=True)
        files = sorted(item.pop("files"), key=lambda f: f["size_bytes"], reverse=True)
        return {"artifact": item, **page(files, offset, limit)}

    @api.post("/api/v1/storage/cleanup/preview", tags=["存储管理"])
    def storage_preview(r: CleanupPreviewRequest):
        return m.storage.preview(r)

    @api.post("/api/v1/storage/cleanup", status_code=202, tags=["存储管理"])
    def storage_cleanup(r: CleanupRequest):
        return m.storage.execute(r)

    @api.get("/api/v1/storage/cleanup/{id}", tags=["存储管理"])
    def storage_cleanup_status(id: str):
        return db.get(id, "storage_cleanup")

    @api.get("/api/v1/system", tags=["环境"])
    def system():
        d = shutil.disk_usage(db.root)
        return {
            "version": VERSION,
            "python": sys.version,
            "python_executable": sys.executable,
            "model_path": m.model_path,
            "data_dir": str(db.root),
            "disk": {
                "total_bytes": d.total,
                "used_bytes": d.used,
                "free_bytes": d.free,
            },
            "dependencies": {
                n: version(n)
                for n in (
                    "verl",
                    "torch",
                    "vllm",
                    "transformers",
                    "peft",
                    "pyarrow",
                    "fastapi",
                )
            },
        }

    @api.get("/api/v1/capabilities", tags=["环境"])
    def capabilities():
        p = m.probe()
        return {
            "task_types": ["cpt", "sft"],
            "finetune": ["full", "lora"],
            "engines": ["fsdp", "fsdp2"],
            "multi_node": False,
            "text_only": True,
            "thinking_token_budget": p["thinking_token_budget"],
            "vllm_available": bool(p["vllm_help"]),
            "lora": {
                "rank": True,
                "alpha": True,
                "target_modules": True,
                "dropout": False,
                "rs_lora": False,
            },
            "versions": p["versions"],
            "verl_commit": p["verl_commit"],
            "runtime_env_allowlist": sorted(ENV_ALLOW),
        }

    @api.get("/api/v1/config/defaults", tags=["环境"])
    def defaults():
        return {
            "verl": m.probe()["defaults"],
            "train_request": TrainRequest.model_json_schema(),
            "prepare_request": PrepareRequest.model_json_schema(),
            "inference_request": InferenceRequest.model_json_schema(),
            "chat_request": ChatRequest.model_json_schema(),
        }

    @api.get("/api/v1/settings", tags=["环境"])
    def get_settings():
        return settings(db)

    @api.patch("/api/v1/settings", tags=["环境"])
    def change_settings(r: SettingsRequest):
        with m.lock:
            v = {**settings(db), **r.model_dump(exclude_none=True)}
            atomic(db.root / "settings.json", v)
            return v

    @api.get("/api/v1/gpus", tags=["GPU"])
    def gpus():
        return m.gpus()

    @api.get("/api/v1/gpus/events", tags=["GPU"])
    async def gpu_events(request: Request):
        async def stream():
            while not await request.is_disconnected():
                yield (
                    "event: gpu\ndata: " + js(await asyncio.to_thread(m.gpus)) + "\n\n"
                )
                await asyncio.sleep(settings(db)["gpu_poll_seconds"])

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post("/api/v1/datasets", tags=["数据"])
    def create_dataset(r: DatasetRequest):
        return db.add("dataset", {**r.model_dump(), "sample_count": 0})

    @api.post("/api/v1/datasets/demo", tags=["数据"])
    def demo(r: DemoRequest):
        with m.lock:
            check(not m.storage.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            ds = db.add(
                "dataset",
                {
                    "name": r.task_type + "-demo",
                    "task_type": r.task_type,
                    "sample_count": 0,
                },
            )
            data = [
                {
                    "messages": [
                        {"role": "user", "content": f"{i}+1等于几？"},
                        {"role": "assistant", "content": f"{i}+1等于{i + 1}。"},
                    ],
                    "source_id": f"demo-{i}",
                    "enable_thinking": False,
                }
                if r.task_type == "sft"
                else {
                    "text": f"这是第{i + 1}份演示资料。项目编号DEMO-{i:03d}，用于检查文本训练流程，不代表真实业务资料。",
                    "source_id": f"demo-{i}",
                }
                for i in range(16)
            ]
            return append(db, ds["id"], data)

    @api.get("/api/v1/datasets", tags=["数据"])
    def list_datasets(
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return page(db.all("dataset"), offset, limit)

    @api.get("/api/v1/datasets/{id}", tags=["数据"])
    def get_dataset(id: str):
        return db.get(id, "dataset")

    @api.post("/api/v1/datasets/{id}/samples", tags=["数据"])
    def add_samples(id: str, r: SamplesRequest):
        with m.lock:
            check(not m.storage.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            check(db.get(id, "dataset")["state"] != "deleted", "数据集已删除或不完整", "ARTIFACT_UNAVAILABLE", 409)
            return append(db, id, r.samples)

    @api.get("/api/v1/datasets/{id}/samples", tags=["数据"])
    def get_samples(
        id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=200)
    ):
        ds = db.get(id, "dataset")
        p = db.root / "datasets" / id / "raw.jsonl"
        return {
            "items": list(itertools.islice(rows(p), offset, offset + limit))
            if p.exists()
            else [],
            "total": ds["sample_count"],
            "offset": offset,
            "limit": limit,
        }

    def edit(id, index, value):
        with m.lock:
            check(not m.storage.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            check(db.get(id, "dataset")["state"] != "deleted", "数据集已删除或不完整", "ARTIFACT_UNAVAILABLE", 409)
            ds = db.get(id, "dataset")
            check(0 <= index < ds["sample_count"], "样本索引不存在", "NOT_FOUND", 404)
            if value is not None:
                value = normalize(value, ds["task_type"])
            with dslock(db, id) as p:
                temp = p / "edit.tmp"
                count = 0
                with temp.open("w", encoding="utf-8") as out:
                    for i, row in enumerate(rows(p / "raw.jsonl")):
                        if i == index:
                            row = value
                        if row is not None:
                            out.write(js(row) + "\n")
                            count += 1
                temp.replace(p / "raw.jsonl")
                return db.update(id, sample_count=count)

    @api.put("/api/v1/datasets/{id}/samples/{index}", tags=["数据"])
    def edit_sample(id: str, index: int, r: EditRequest):
        return edit(id, index, r.sample)

    @api.delete("/api/v1/datasets/{id}/samples/{index}", tags=["数据"])
    def delete_sample(id: str, index: int):
        return edit(id, index, None)

    def import_check(id, ext):
        ds = db.get(id, "dataset")
        check(ds["state"] != "deleted", "数据集已删除或删除未完成", "ARTIFACT_UNAVAILABLE", 409)
        check(
            ext in (".txt", ".md", ".json", ".jsonl", ".parquet"),
            "不支持文件格式",
            "UNSUPPORTED_FORMAT",
            422,
        )
        check(
            ds["task_type"] == "cpt" or ext not in (".txt", ".md"),
            "SFT需问答数据",
            "INVALID_DATA",
            422,
        )

    @api.post("/api/v1/datasets/{id}/upload", status_code=202, tags=["数据"])
    async def upload(id: str, file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        import_check(id, ext)
        p = db.root / "uploads" / (ident("upload") + ext)
        p.parent.mkdir(exist_ok=True)
        size = 0
        try:
            with p.open("wb") as out:
                while b := await file.read(1024**2):
                    size += len(b)
                    check(
                        size <= settings(db)["max_upload_bytes"],
                        "文件过大",
                        "UPLOAD_TOO_LARGE",
                        413,
                    )
                    await asyncio.to_thread(out.write, b)
        except BaseException:
            p.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        try:
            return public(m.job("import", {"filename": file.filename}, dataset_id=id, snapshot=str(p)))
        except BaseException:
            p.unlink(missing_ok=True)
            raise

    @api.post("/api/v1/datasets/{id}/import", status_code=202, tags=["数据"])
    def import_file(id: str, r: ImportRequest):
        p = Path(r.path).expanduser().resolve()
        check(p.is_file(), "文件不存在", "NOT_FOUND", 404)
        import_check(id, p.suffix.lower())
        check(
            p.stat().st_size <= settings(db)["max_upload_bytes"],
            "文件过大",
            "UPLOAD_TOO_LARGE",
            413,
        )
        dest = db.root / "uploads" / (ident("import") + p.suffix.lower())
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, dest)
        return public(
            m.job("import", r.model_dump(), dataset_id=id, snapshot=str(dest))
        )

    @api.post("/api/v1/datasets/{id}/prepare", status_code=202, tags=["数据"])
    def prepare(id: str, r: PrepareRequest):
        with m.lock:
            check(not m.storage.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            check(db.get(id, "dataset")["state"] != "deleted", "数据集已删除或不完整", "ARTIFACT_UNAVAILABLE", 409)
            ds = db.get(id, "dataset")
            db.get(r.model_id, "model")
            check(ds["sample_count"] > 0, "数据集为空", "EMPTY_DATASET", 422)
            dest = db.root / "snapshots" / (ident("snapshot") + ".jsonl")
            dest.parent.mkdir(exist_ok=True)
            with dslock(db, id) as p:
                shutil.copyfile(p / "raw.jsonl", dest)
            return public(
                m.job(
                    "prepare",
                    r.model_dump(),
                    dataset_id=id,
                    snapshot=str(dest),
                    version_id=ident("dv"),
                )
            )

    @api.get("/api/v1/dataset-versions", tags=["数据"])
    def dataset_versions(
        offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        return page(db.all("dataset_version"), offset, limit)

    @api.get("/api/v1/dataset-versions/{id}", tags=["数据"])
    def dataset_version(id: str):
        return db.get(id, "dataset_version")

    @api.get("/api/v1/dataset-versions/{id}/preview", tags=["数据"])
    def preview(
        id: str,
        split: Literal["train", "val"] = "train",
        offset: int = Query(0, ge=0),
        limit: int = Query(5, ge=1, le=20),
    ):
        import pyarrow.parquet as pq

        dv = db.get(id, "dataset_version")
        m.storage.available(dv)
        tok = tokenizer(dv["tokenizer_path"])
        result = []
        i = 0
        for batch in pq.ParquetFile(Path(dv["path"]) / f"{split}.parquet").iter_batches(
            batch_size=64
        ):
            for row in batch.to_pylist():
                if i < offset:
                    i += 1
                    continue
                row["decoded_text"] = tok.decode(
                    row["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                row["supervised_text"] = tok.decode(
                    [t for t, mask in zip(row["input_ids"], row["loss_mask"]) if mask],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                result.append(row)
                if len(result) >= limit:
                    break
            if len(result) >= limit:
                break
        return {
            "items": result,
            "total": dv["report"][split + "_samples"],
            "offset": offset,
            "limit": limit,
        }

    @api.get("/api/v1/dataset-versions/{id}/files/{name}", tags=["数据"])
    def data_file(id: str, name: str):
        check(
            name in ("train.parquet", "val.parquet", "manifest.json", "report.json"),
            "未知文件",
            "NOT_FOUND",
            404,
        )
        return FileResponse(
            safe_file(db.get(id, "dataset_version")["path"], name), filename=name
        )

    @api.post("/api/v1/jobs/validate", tags=["训练"])
    def validate(r: TrainRequest):
        return m.validate(r)

    @api.post("/api/v1/jobs", status_code=202, tags=["训练"])
    def train(r: TrainRequest):
        return public(m.train(r, m.validate(r)))

    @api.get("/api/v1/jobs", tags=["任务"])
    def jobs(
        kind: str | None = None,
        state: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return page(
            [
                public(j)
                for j in db.all("job")
                if (not kind or j["job_type"] == kind)
                and (not state or j["state"] == state)
            ],
            offset,
            limit,
        )

    @api.get("/api/v1/jobs/{id}", tags=["任务"])
    def get_job(id: str):
        return public(db.get(id, "job"))

    @api.post("/api/v1/jobs/{id}/cancel", tags=["任务"])
    def cancel(id: str):
        return public(m.cancel(id))

    @api.post("/api/v1/jobs/{id}/resume", status_code=202, tags=["训练"])
    def resume(id: str, r: ResumeRequest):
        original = db.get(id, "job")
        check(
            original["job_type"] == "train" and original["state"] in DONE,
            "仅可恢复已结束训练",
            "STATE_CONFLICT",
            409,
        )
        ck = db.get(r.checkpoint_id, "checkpoint")
        check(
            ck["job_id"] == id and ck.get("resumable") and Path(ck["path"]).is_dir(),
            "检查点不属于本任务或不可恢复",
        )
        body = dict(original["request"])
        body.update(
            epochs=int(original["training_config"]["overrides"]["trainer.total_epochs"])
            + r.additional_epochs,
            max_steps=None,
            name=body["name"] + "-resume",
        )
        body["verl_overrides"] = {
            k: v
            for k, v in body["verl_overrides"].items()
            if k not in ("trainer.total_epochs", "trainer.total_training_steps")
        }
        request = TrainRequest.model_validate(body)
        cfg = m.validate(request, ck["path"])
        check(len(cfg["gpu_uuids"]) == ck["world_size"], "恢复需原GPU数量")
        return public(m.train(request, cfg, resumed_from=ck["id"]))

    @api.get("/api/v1/jobs/{id}/metrics", tags=["监控"])
    def metrics(
        id: str, after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)
    ):
        db.get(id, "job")
        items = db.metric_rows(id, after, limit)
        return {"items": items, "next_after": items[-1]["seq"] if items else after}

    @api.get("/api/v1/jobs/{id}/logs", tags=["监控"])
    def logs(
        id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(65536, ge=4, le=1048576),
    ):
        p = Path(db.get(id, "job")["log_path"])
        if not p.exists():
            return {"text": "", "next_offset": 0, "size": 0, "eof": True}
        size = p.stat().st_size
        check(offset <= size, "offset超过文件大小", status=422)
        with p.open("rb") as f:
            f.seek(offset)
            data = f.read(limit)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        text = decoder.decode(data, final=offset + len(data) >= size)
        end = offset + len(data) - len(decoder.getstate()[0])
        return {"text": text, "next_offset": end, "size": size, "eof": end >= size}

    async def events_response(request, after, ref=None):
        h = request.headers.get("last-event-id")
        if h:
            check(h.isdigit(), "Last-Event-ID需整数", status=422)
            after = max(after, int(h))

        async def stream():
            cursor = after
            heartbeat = time.monotonic()
            with db.connect() as c:
                minimum = c.execute("SELECT MIN(seq) FROM events").fetchone()[0] or 0
            if cursor and cursor < minimum - 1:
                cursor = minimum - 1
                yield (
                    "event: reset\ndata: "
                    + js(
                        {
                            "next_after": cursor,
                            "reason": "事件窗口已过，请从metrics/logs补读",
                        }
                    )
                    + "\n\n"
                )
            while not await request.is_disconnected():
                for event in await asyncio.to_thread(db.events, cursor, ref):
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: {event['kind']}\ndata: {js(event)}\n\n"
                if time.monotonic() - heartbeat >= 15:
                    yield (
                        "event: heartbeat\ndata: "
                        + js({"timestamp": stamp(), "after": cursor})
                        + "\n\n"
                    )
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/api/v1/jobs/{id}/events", tags=["监控"])
    async def job_events(request: Request, id: str, after: int = Query(0, ge=0)):
        db.get(id, "job")
        return await events_response(request, after, id)

    @api.get("/api/v1/events", tags=["监控"])
    async def events(request: Request, after: int = Query(0, ge=0)):
        return await events_response(request, after)

    @api.get("/api/v1/checkpoints", tags=["版本"])
    def checkpoints(
        job_id: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return page(
            [c for c in db.all("checkpoint") if not job_id or c["job_id"] == job_id],
            offset,
            limit,
        )

    @api.post("/api/v1/exports", status_code=202, tags=["版本"])
    def export(r: ExportRequest):
        ck = db.get(r.checkpoint_id, "checkpoint")
        check(
            ck["state"] != "missing"
            and (Path(ck["path"]) / "verl_api_complete.json").is_file(),
            "检查点不可用",
        )
        return public(m.job("export", r.model_dump()))

    @api.get("/api/v1/models", tags=["版本"])
    def models(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return page(db.all("model"), offset, limit)

    @api.post("/api/v1/models/register", tags=["版本"])
    def register(r: RegisterRequest):
        with m.lock:
            check(not m.storage.cleaning, "清理进行中", "STORAGE_BUSY", 409)
            p = str(Path(r.path).expanduser().resolve())
            verify_model(p, r.model_kind)
            tp = p
            if r.model_kind == "lora":
                check(r.base_model_id, "adapter需base_model_id")
                base = db.get(r.base_model_id, "model")
                check(base["model_kind"] == "full", "底座需完整模型")
                tp = base.get("tokenizer_path", base["path"])
            return db.add(
                "model",
                {**r.model_dump(), "path": p, "tokenizer_path": tp, "chat_defaults": {}},
            )

    @api.patch("/api/v1/models/{id}/chat-defaults", tags=["对话"])
    def chat_defaults(id: str, r: ChatOptions):
        model = db.get(id, "model")
        return db.update(
            id,
            chat_defaults={
                **model.get("chat_defaults", {}),
                **r.model_dump(exclude_unset=True),
            },
        )

    @api.get("/api/v1/models/{id}/files", tags=["版本"])
    def model_files(id: str):
        model = db.get(id, "model")
        m.storage.available(model)
        p = Path(model["path"])
        return {
            "model_id": id,
            "items": [
                {"path": str(f.relative_to(p)), "size_bytes": f.stat().st_size}
                for f in p.rglob("*")
                if f.is_file()
            ],
        }

    @api.get("/api/v1/models/{id}/files/{relative:path}", tags=["版本"])
    def model_file(id: str, relative: str):
        p = safe_file(db.get(id, "model")["path"], relative)
        return FileResponse(p, filename=p.name)

    @api.post("/api/v1/inference", status_code=202, tags=["推理"])
    def load_inference(r: InferenceRequest):
        model = db.get(r.model_id, "model")
        m.storage.available(model)
        verify_model(model["path"], model["model_kind"])
        check(r.tensor_parallel_size == len(r.gpu_ids), "TP大小应等于选定GPU数")
        check(
            r.max_lora_rank in (1, 8, 16, 32, 64, 128, 256, 512),
            "max_lora_rank需1/8/16/32/64/128/256/512",
        )
        environment = m.probe()
        check(environment["vllm_help"], "未发现可用vLLM", "VLLM_UNAVAILABLE", 503)
        for k in r.extra_args:
            check(k.replace("_", "-") in VLLM_EXTRA, f"未开放vLLM参数:{k}")
        return public(
            m.job(
                "inference",
                r.model_dump(),
                gpu_uuids=select_gpus(r.gpu_ids, m.gpus(True)),
            )
        )

    @api.get("/api/v1/inference", tags=["推理"])
    def instances():
        return page(
            [public(j) for j in db.all("job") if j["job_type"] == "inference"],
            0,
            100000,
        )

    @api.delete("/api/v1/inference/{id}", tags=["推理"])
    def unload(id: str):
        with m.lock:
            check(db.get(id, "job")["job_type"] == "inference", "不是推理实例")
            check(
                not m.chat_counts.get(id, 0),
                "实例正在生成或加载adapter",
                "INFERENCE_BUSY",
                409,
            )
            return public(m.cancel(id))

    def get_instance(id):
        j = db.get(id, "job")
        check(
            j["job_type"] == "inference" and j["state"] == "ready",
            "推理实例未ready",
            "MODEL_NOT_READY",
            409,
        )
        return j

    @api.post("/api/v1/inference/{id}/adapters", tags=["推理"])
    def add_adapter(id: str, r: AdapterRequest):
        with m.lock:
            inst = get_instance(id)
            model = db.get(r.model_id, "model")
            m.storage.admit(r.model_dump())
            check(
                model["model_kind"] == "lora"
                and model["base_model_id"] == inst["base_model_id"],
                "adapter底座不匹配",
            )
            ac = json.loads((Path(model["path"]) / "adapter_config.json").read_text())
            check(ac["r"] <= inst["request"]["max_lora_rank"], "rank超过实例限制")
            m.chat_counts[id] = m.chat_counts.get(id, 0) + 1
        try:
            with httpx.Client(timeout=120, trust_env=False) as c:
                response = c.post(
                    inst["internal_url"] + "/v1/load_lora_adapter",
                    headers={"Authorization": "Bearer " + inst["internal_key"]},
                    json={"lora_name": model["id"], "lora_path": model["path"]},
                )
            check(response.is_success, response.text[:3000], "ADAPTER_LOAD_FAILED", 503)
            with m.lock:
                return public(
                    db.update(
                        id,
                        served_models=sorted(
                            set(db.get(id)["served_models"] + [model["id"]])
                        ),
                    )
                )
        finally:
            with m.lock:
                m.chat_counts[id] = max(0, m.chat_counts.get(id, 1) - 1)

    @api.delete("/api/v1/inference/{id}/adapters/{model_id}", tags=["推理"])
    def del_adapter(id: str, model_id: str):
        with m.lock:
            inst = get_instance(id)
            check(
                model_id in inst["served_models"]
                and model_id
                not in (inst["base_model_id"], inst["request"]["model_id"]),
                "只能卸载动态加载的adapter",
            )
            check(not m.chat_counts.get(id, 0), "实例忙", "INFERENCE_BUSY", 409)
            m.chat_counts[id] = 1
        try:
            with httpx.Client(timeout=30, trust_env=False) as c:
                response = c.post(
                    inst["internal_url"] + "/v1/unload_lora_adapter",
                    headers={"Authorization": "Bearer " + inst["internal_key"]},
                    json={"lora_name": model_id},
                )
            check(
                response.is_success, response.text[:3000], "ADAPTER_UNLOAD_FAILED", 503
            )
            with m.lock:
                return public(
                    db.update(
                        id,
                        served_models=[
                            v for v in db.get(id)["served_models"] if v != model_id
                        ],
                    )
                )
        finally:
            with m.lock:
                m.chat_counts[id] = max(0, m.chat_counts.get(id, 1) - 1)

    @api.get("/v1/models", tags=["对话"])
    def loaded_models():
        ids = sorted(
            {
                v
                for j in db.all("job")
                if j["job_type"] == "inference" and j["state"] == "ready"
                for v in j.get("served_models", [])
            }
        )
        return {
            "object": "list",
            "data": [{"id": v, "object": "model", "owned_by": "verl-api"} for v in ids],
        }

    @api.get("/api/v1/chat/requests/{id}", tags=["对话"])
    def chat_status(id: str):
        return db.get(id, "chat")

    @api.post("/api/v1/chat/requests/{id}/cancel", tags=["对话"])
    async def cancel_chat(id: str):
        obj = db.get(id, "chat")
        task = m.chat_tasks.get(id)
        if task:
            task.cancel()
            return {"request_id": id, "state": "stopping"}
        return {"request_id": id, "state": obj["state"]}

    @api.post("/v1/chat/completions", tags=["对话"])
    async def chat(r: ChatRequest, request: Request):
        model = db.get(r.model, "model")
        environment = await asyncio.to_thread(m.probe)
        params = chat_params(
            r, model.get("chat_defaults", {}), environment["thinking_token_budget"]
        )
        with m.lock:
            candidates = [
                j
                for j in db.all("job")
                if j["job_type"] == "inference"
                and j["state"] == "ready"
                and r.model in j.get("served_models", [])
                and (not r.instance_id or j["id"] == r.instance_id)
            ]
            check(
                candidates,
                "请先加载模型并等待推理实例ready；GPU可能正在训练",
                "MODEL_NOT_READY",
                409,
            )
            inst = min(candidates, key=lambda j: m.chat_counts.get(j["id"], 0))
            check(
                params["max_tokens"] < inst["request"]["max_model_len"],
                "max_tokens需小于实例上下文上限",
                "CONTEXT_LIMIT",
                422,
            )
            m.chat_counts[inst["id"]] = m.chat_counts.get(inst["id"], 0) + 1
        id = ident("chat")
        started = time.monotonic()
        db.add(
            "chat",
            {
                "model_id": r.model,
                "instance_id": inst["id"],
                "effective_parameters": params,
                "timing": None,
                "started_at": stamp(),
            },
            state="running",
            id=id,
        )
        body = {
            k: v for k, v in params.items() if k != "enable_thinking" and v is not None
        }
        body.update(
            model=r.model,
            messages=[msg.model_dump() for msg in r.messages],
            stream=r.stream,
            chat_template_kwargs={"enable_thinking": params["enable_thinking"]},
        )
        if r.stream:
            body["stream_options"] = {"include_usage": True}
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=30), trust_env=False
        )
        response = None
        released = False

        def timing(first=None, usage=None):
            elapsed = time.monotonic() - started
            tokens = (usage or {}).get("completion_tokens")
            return {
                "elapsed_seconds": elapsed,
                "time_to_first_token_seconds": first,
                "output_tokens_per_second": (tokens - 1) / max(elapsed - first, 1e-6)
                if first is not None and tokens and tokens > 1
                else None,
            }

        async def finish(status, stats=None, usage=None, error=None):
            nonlocal released
            if released:
                return
            released = True
            try:
                if response is not None:
                    with contextlib.suppress(Exception):
                        await response.aclose()
                await client.aclose()
            finally:
                with m.lock:
                    m.chat_counts[inst["id"]] = max(
                        0, m.chat_counts.get(inst["id"], 1) - 1
                    )
                m.chat_tasks.pop(id, None)
                db.update(
                    id,
                    state=status,
                    finished_at=stamp(),
                    timing=stats,
                    usage=usage,
                    error=error,
                )

        try:
            m.chat_tasks[id] = asyncio.current_task()
            up = client.build_request(
                "POST",
                inst["internal_url"] + "/v1/chat/completions",
                headers={"Authorization": "Bearer " + inst["internal_key"]},
                json=body,
            )
            response = await client.send(up, stream=r.stream)
            if not response.is_success:
                raise Error(
                    (await response.aread()).decode(errors="replace")[:5000],
                    "INFERENCE_ERROR",
                    response.status_code if response.status_code < 500 else 503,
                )
            if not r.stream:
                obj = reasoning(response.json())
                check(
                    not obj.get("error"), js(obj.get("error")), "INFERENCE_ERROR", 503
                )
                stats = timing(usage=obj.get("usage"))
                obj.update(request_id=id, effective_parameters=params, timing=stats)
                await finish("succeeded", stats, obj.get("usage"))
                return JSONResponse(obj, headers={"X-Request-ID": id})
        except BaseException as e:
            await asyncio.shield(
                finish(
                    "cancelled" if isinstance(e, asyncio.CancelledError) else "failed",
                    timing(),
                    error=str(e),
                )
            )
            raise

        async def stream():
            first = None
            usage = None
            status = "succeeded"
            error = None
            got_done = False
            m.chat_tasks[id] = asyncio.current_task()
            try:
                yield (
                    "data: "
                    + js(
                        {
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "request_id": id,
                            "effective_parameters": params,
                        }
                    )
                    + "\n\n"
                )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    text = line[5:].strip()
                    if text == "[DONE]":
                        got_done = True
                        break
                    obj = reasoning(json.loads(text))
                    if obj.get("error"):
                        status = "failed"
                        error = obj["error"]
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        if first is None and (
                            delta.get("content") or delta.get("reasoning_content")
                        ):
                            first = time.monotonic() - started
                    if obj.get("usage"):
                        usage = obj["usage"]
                    obj["request_id"] = id
                    yield "data: " + js(obj) + "\n\n"
                if not got_done and status != "failed":
                    raise Error(
                        "上游流提前断开，未收到[DONE]", "STREAM_INCOMPLETE", 503
                    )
                yield (
                    "data: "
                    + js(
                        {
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "request_id": id,
                            "timing": timing(first, usage),
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as e:
                status = "failed"
                error = str(e)
                yield (
                    "data: "
                    + js(
                        {
                            "error": {
                                "code": getattr(e, "code", "STREAM_FAILED"),
                                "message": str(e),
                            },
                            "request_id": id,
                        }
                    )
                    + "\n\ndata: [DONE]\n\n"
                )
            finally:
                await asyncio.shield(finish(status, timing(first, usage), usage, error))

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": id,
            },
        )

    app.include_router(api)
    return app


def main():
    mode = os.environ.get("VERL_API_MODE", "serve")
    if mode == "rank":
        train_rank()
        return
    if mode == "worker":
        worker()
        return
    if mode == "probe":
        atomic(os.environ["VERL_API_OUTPUT"], probe())
        return
    if mode == "validate":
        from omegaconf import OmegaConf

        cfg = compose(json.loads(Path(os.environ["VERL_API_INPUT"]).read_text()))
        atomic(os.environ["VERL_API_OUTPUT"], OmegaConf.to_container(cfg, resolve=True))
        return
    parser = argparse.ArgumentParser(
        description="容器内verl API，仅三个位置参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("service_ip", nargs="?", default="0.0.0.0", help="监听IP")
    parser.add_argument(
        "service_port", nargs="?", type=int, default=7877, help="监听端口"
    )
    parser.add_argument(
        "model_path", nargs="?", default=MODEL_DEFAULT, help="训练起点模型的容器内路径"
    )
    a = parser.parse_args()
    try:
        ipaddress.ip_address(a.service_ip)
        check(1 <= a.service_port <= 65535, "端口需1..65535")
        verify_model(a.model_path, "full")
        import uvicorn

        app = create_app(a.model_path)
        print(
            f"Verl API {VERSION}: http://{a.service_ip}:{a.service_port}/docs",
            flush=True,
        )
        print(
            f"API Key文件: {ROOT / 'api_key'}；完整规范/api-spec；训练监控/jobs/{{id}}/metrics、logs、events",
            flush=True,
        )
        uvicorn.run(app, host=a.service_ip, port=a.service_port, workers=1)
    except (Error, ValueError) as e:
        parser.exit(2, f"启动失败:{e}\n")


if __name__ == "__main__":
    main()
