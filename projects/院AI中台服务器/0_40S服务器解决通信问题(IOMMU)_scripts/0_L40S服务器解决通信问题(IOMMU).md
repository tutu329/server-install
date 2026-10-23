# 四卡 L40S 服务器 IOMMU 修改与测试说明

版本 1.0    记录日期 2026年10月22日    对象 主机 zs 的管理员

本次通过在 GRUB 内核启动参数中添加 intel_iommu=off 并重启，恢复了四卡服务器的 GPU 间数据传输正确性。修复后 12 个有向 GPU 传输组合全部通过，用户已确认 LlamaFactory WebUI 的 Chat 可以正常对话。

这台服务器已经完成修改。日常检查从第 4 节开始，无需再次创建配置文件。本文记录故障证据、修改步骤、验收测试、回滚方式及后续性能测试方法；所有 sudo、GRUB、journalctl 命令都在宿主机执行。

## 1 已确认的服务器配置

| 项目 | 本次实际环境 |
| --- | --- |
| 主机与内核 | zs；Ubuntu；6.8.0-139-generic；裸机检测结果 none |
| 主板与 BIOS | inagile NF5468M6；AMI 08.04.00；2023-12-08 |
| CPU | 2 × Intel Xeon Gold 5318Y；共 48 核 96 线程 |
| GPU 与驱动 | 4 × NVIDIA L40S；每卡 46068 MiB；驱动 610.57.04 |
| 容器与推理环境 | llamafactory-webui；hiyouga/llamafactory:latest
PyTorch 2.6.0+cu124；PyTorch CUDA 12.4 |
| 故障时模型 | Qwen3.5-0.8B；Hugging Face 推理；qwen3_5 模板 |

latest 标签会变化；复测时应记录实际镜像 ID 和已安装包版本。nvidia-smi 中 CUDA UMD 13.3 与 PyTorch 自带的 CUDA 12.4 不是同一个版本字段，不能混用。

| GPU | PCI 地址 | NUMA | 同组连接 |
| --- | --- | --- | --- |
| 0 | 0000:35:00.0 | 0 | 与 GPU 1 为 PIX |
| 1 | 0000:37:00.0 | 0 | 与 GPU 0 为 PIX |
| 2 | 0000:9b:00.0 | 1 | 与 GPU 3 为 PIX |
| 3 | 0000:9d:00.0 | 1 | 与 GPU 2 为 PIX |

GPU 0/1 与 GPU 2/3 之间为 SYS，通信经过 CPU 间互联。NUMA 0 的 CPU 为 0–23、48–71，NUMA 1 为 24–47、72–95。重启或设备调整后，以 PCI 地址和 UUID 重新核对编号。

## 2 故障证据与结论边界

最初训练可运行，Chat 在 embedding 查表附近触发 CUDA 设备断言。单卡推理正常，自动分配到多卡时出错；故障时语言模型和 lm_head 位于 GPU 3，视觉模块位于 GPU 1。

```
Assertion `srcIndex < srcSelectDimSize` failed
RuntimeError: CUDA error: device-side assert triggered
Device map: {'lm_head': 3,
             'model.language_model': 3, 'model.visual': 1}
```

输入索引越界是表面症状，仅凭这段异常不能认定硬件损坏或词表不匹配。模型 embedding 行数为 248320，tokenizer 长度为 248077；这个大小关系本身正常。随后检查发现，故障还可以在独立的整数跨卡复制测试中复现。

| 验证项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 4 张卡单卡往返与整数计算 | 全部 PASS | 全部 PASS |
| GPU 0/1 向其余卡发送 | 6 个方向存在错误 | 6 个方向全部 PASS |
| GPU 2/3 向其余卡发送 | 6 个方向全部 PASS | 6 个方向全部 PASS |
| 显式经 CPU 中转复制 | 全部通过 | 全部通过 |
| 跨卡复制失败子测试数 | 47 | 0 |
| LlamaFactory Chat | 设备断言失败 | 用户确认成功 |

47 表示不同长度和重复轮次中的失败子测试数，不是 47 个传输方向。四张卡只有 4 × 3 = 12 个有向组合。P2P 能力查询在修改前也返回可访问，说明“支持访问”不能代替实际数据校验。

```
DMAR: [DMA Write NO_PASID] Request device [37:00.0]
      [fault reason 0x05] PTE Write access is not set
# 35:00.0 也出现同类错误；对应 GPU 0 和 GPU 1。
```

这些日志说明 DMA 写访问遭到 IOMMU 页表权限拒绝。它与 GPU 0/1 作为发送端时的数据错误、CPU 中转正常，以及配置修改后的恢复相互吻合，支持问题位于服务器 IOMMU/PCIe 访问路径配置。GPU 3 的 Xid 43 出现在应用失败时，不能据此认定 GPU 3 已损坏。

修改前相关 PCIe 桥的 ACS 请求及完成重定向处于开启状态；重启后观察到 ReqRedir-、CmpltRedir-。IOMMU 与 ACS 状态在同一次重启后共同变化，本次没有做独立变量实验，因此不能量化两者各自的贡献，也不能据此宣布整机硬件已完成全面健康认证。

## 3 修改 IOMMU 的操作记录

### 3.1 适用条件和维护准备

本方案适用于本机这种直接运行 CUDA 的 Intel 裸机环境。NVIDIA 文档说明 Linux 裸机的 IOMMU 启用状态与 PCIe P2P 传输存在支持限制［1］。这不是对所有服务器统一关闭 IOMMU 的建议。

关闭 IOMMU 会减少设备 DMA 地址隔离保护，并影响依赖 IOMMU 的 VFIO 设备直通或相关虚拟化隔离。普通 Docker GPU 使用与 VFIO 直通不同；若以后部署设备直通虚拟机或不可信多租户任务，应重新评估。本操作不等于关闭 CPU VT-x，也不要求关闭 IRQ remapping。

维护前结束训练、推理等任务，确认可以重启并具备 SSH 或带外控制台恢复通道。先保存诊断日志；不要删除原容器或模型缓存。下面是本次已成功执行的修改脚本，当前服务器无需重跑。

```
sudo bash <<'SH'
set -eu
cfg=/etc/default/grub.d/99-l40s-p2p.cfg
if [ -e "$cfg" ]; then
    echo "配置文件已存在，停止操作，请先查看：$cfg"
    exit 1
fi
mkdir -p /etc/default/grub.d
cat > "$cfg" <<'EOF'
# Disable Intel IOMMU for bare-metal CUDA PCIe P2P.
GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} intel_iommu=off"
EOF
update-grub
echo "确认以下启动项包含 intel_iommu=off："
grep -E '^[[:space:]]*linux.*intel_iommu=off' /boot/grub/grub.cfg
SH
```

此脚本保留原 GRUB_CMDLINE_LINUX 内容，新增独立配置文件，便于回滚。带引号的 EOF 保留变量原文，供 update-grub 加载时展开。不要直接编辑自动生成的 /boot/grub/grub.cfg。

### 3.2 确认生成成功后重启

本次 update-grub 成功找到 6.8.0-139 和 6.8.0-134 内核，生成的 linux 启动行均包含 intel_iommu=off。os-prober 警告仅说明不扫描其他操作系统，不代表这次 GRUB 生成失败。

```
sudo reboot
```

参数必须重启后才生效。Docker 容器内部添加这个参数或只重启容器，不能替代宿主机内核设置。

## 4 重启后的状态检查

```
cat /proc/cmdline
sudo cat /etc/default/grub.d/99-l40s-p2p.cfg
sudo journalctl -k -b --no-pager | \
  grep -Ei 'DMAR|IOMMU|DMA.*fault'
nvidia-smi
nvidia-smi topo -m
```

本次已确认启动参数包含 intel_iommu=off，当前启动日志明确出现 DMAR: IOMMU disabled。四张卡可识别，后续跨卡复制与 Chat 验证通过。

| 日志或现象 | 正确解读 |
| --- | --- |
| DMAR: IOMMU disabled | Intel IOMMU 已被禁用，是本次检查关键。 |
| ACPI: DMAR、DRHD、RMRR 等 | 固件表和硬件描述仍可打印，不代表关闭失败。 |
| DMAR-IR: Enabled IRQ remapping | 中断重映射与 DMA 地址重映射不同，不需要为了消除这行日志继续修改。 |
| Default domain type: Translated | 通用 IOMMU 框架默认策略提示，不能单独证明 Intel DMA 重映射仍在运行。 |
| _OSC: platform does not support AER | 平台能力协商信息，不能当作实际 PCIe Bus Error；也不代表所有硬件故障都能由 OS AER 捕获。 |

### 4.1 在复测窗口内检查新增错误

在同一个宿主机终端执行下行，记下复测开始时间；完成第 5 节后再执行后续日志查询。使用 -b 只查当前启动，避免把旧故障日志当成新错误。

```
test_start=$(date --iso-8601=seconds)
```

```
sudo journalctl -k -b --since "$test_start" --no-pager | \
  grep -Ei 'DMA.*fault|PTE Write access|NVRM: Xid|PCIe Bus Error|AER:.*error'
```

理想结果是没有输出。grep 没有匹配行时退出码为 1，在这里通常表示未找到所筛选的错误，而不是测试失败。传输和 Chat 成功仍是必要验收条件；只看启动日志不足以确认修复。

### 4.2 ACS 与链路只读复核

```
sudo lspci -Dvv | \
  grep -E '^[0-9a-fA-F]{4}:|LnkSta:|ACSCtl:'
```

本次重启后 GPU 上游相关桥的 ReqRedir、CmpltRedir 已为负号，无需额外运行 setpci。修复后采样显示四卡链路为 Gen4 ×16；空闲时链路速率降档可能是节能行为，应结合负载观察，不能只凭空闲速率判故障。

## 5 复测顺序与验收方法

下载并解压随附完整包，进入 L40S_IOMMU_Guide 目录。脚本使用现有 llamafactory-webui 容器，不会创建或删除容器。测试前确保四卡可见、显存有余量，并避开其他训练或 vLLM 负载；关闭会增加显存占用的重复模型实例。

### 5.1 保存环境并运行传输校验

```
run_dir="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
bash scripts/collect_system.sh llamafactory-webui \
  > "$run_dir/system.txt" 2>&1
bash scripts/run_transfer.sh llamafactory-webui \
  "$run_dir/transfer.txt"
echo "传输测试退出码：$?"
```

每次使用新的日志目录。run_transfer.sh 使用 pipefail，失败不会被 tee 掩盖；退出码 0 才是成功。collect_system.sh 为尽量保留现场会继续收集可用项目，其退出码不能替代逐项查看日志。

校验内容包括每卡 CPU→GPU→CPU 往返、整数运算，以及全部 12 个有向 GPU→GPU 复制组合。每个方向测试 32、4096、1048576 个 int64 元素，各重复 3 轮；每轮更换数据，并与显式经 CPU 中转的路径对照。完整 Python 源码见附录 A。

验收要求：4 张卡单卡检查全部 PASS；12 个方向全部 PASS；出现 All tested copies and integer checks PASSED.；退出码为 0。该脚本验证 PyTorch 实际复制路径的数据正确性，不测带宽，也不独立证明每次复制必然采用直接 PCIe P2P。

### 5.2 验证实际 Chat

打开现有 WebUI，使用原来的模型和模板 qwen3_5，加载模型后连续进行多轮对话。确认没有 CUDA 断言、输出正常结束，再执行第 4 节的内核错误查询。发生过 device-side assert 的旧 Python 进程应退出重建，避免继续使用异常 CUDA 上下文。

若需保留命令行复测日志，可在 WebUI 卸载模型、释放显存后执行随包的 chat_smoke.py。该步骤是额外的模型加载，会占用显存；已有 WebUI 成功结果可作为业务验收。

```
set -o pipefail
sudo docker exec -i \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  llamafactory-webui python -u - \
  < scripts/chat_smoke.py 2>&1 | tee "$run_dir/chat.txt"
```

Chat 脚本默认使用原 ModelScope 缓存路径，路径变化时用 docker exec -e MODEL_PATH=实际路径 传入。CUDA_LAUNCH_BLOCKING=1 仅用于本次诊断进程，不应据此测性能。若自动映射只用了单卡，这次 Chat 只能证明应用可用；跨卡正确性由独立的 12 方向测试覆盖。

## 6 回滚 后续性能测试与维护

### 6.1 回滚本次修改

仅在需要恢复 IOMMU、部署 VFIO 直通或维护策略要求时回滚。先停止跨卡业务；原故障可能重新出现。以下命令只移除本次独立配置，不修改其他 GRUB 参数。

```
sudo cat /etc/default/grub.d/99-l40s-p2p.cfg
sudo rm /etc/default/grub.d/99-l40s-p2p.cfg
sudo update-grub
```

确认 update-grub 成功且生成的目标启动项不再包含本次追加的参数后，再重启。如果其他配置中也有 intel_iommu=off，需先核对来源，不能假定删除这个文件就已全部清除。

```
sudo reboot
# 重启后核对
cat /proc/cmdline
sudo journalctl -k -b --no-pager | grep -Ei 'DMAR|IOMMU'
```

若修改后无法正常启动，可通过带外控制台进入 GRUB，按 e 编辑启动项，临时移除 linux 行末尾的 intel_iommu=off，再按 Ctrl+X 或 F10 启动。进入系统后完成上述持久化回滚。此方式只适用于 GRUB 可访问的情况。

### 6.2 单独复测四卡与两卡性能

当前已证明数据正确性和 Chat 可用性恢复，尚无同条件 vLLM 两卡与四卡性能对照结果。四卡并行通信量更大，且本机跨组走 SYS，因此四卡不一定比同组两卡延迟更低。不能把“四卡较慢”直接等同于硬件故障。

后续按三个层次测试：先用 NVIDIA nvbandwidth 测 GPU 间带宽；再用 NCCL 测试验证集合通信；最后比较实际 vLLM 服务。每轮先确保数据校验通过，保存工具、驱动、镜像和模型版本。工具安装与运行参数以各自官方文档为准［2］［4］［5］。

| 对照组合 | 保持一致的条件 | 记录指标 |
| --- | --- | --- |
| GPU 0/1；GPU 2/3；跨组双卡 | 同一传输大小与工具配置 | 方向带宽、延迟及校验结果 |
| TP=2 同组双卡；TP=4 四卡 | 同一模型、精度、输入集、并发、输出长度；固定预热与重复次数 | tokens/s、TTFT、每 token 延迟、GPU 利用率 |

NCCL_P2P_DISABLE 仅控制 NCCL 的 P2P 传输，不能修复普通 PyTorch 跨卡 .to() 复制。若过去设置过这类绕行参数，应在隔离的测试进程中记录并对照验证，不要在生产任务运行中盲目调整。

### 6.3 后续维护验收

本次记录：内核参数生效；4 卡单卡检查通过；12 个方向通过；提供的重启后日志未见原 DMA fault 或 Xid；用户确认 Chat 成功。驱动、内核、BIOS、容器或 PCIe 插槽调整后，重新执行第 4、5 节。当前不需要为此故障修改模型词表或追加 LlamaFactory 补丁。

## 附录 A 四卡传输测试完整源码

文件 scripts/transfer_check.py，分两页展示。建议直接运行随包文件，避免复制 PDF 时破坏缩进。与故障定位时的检查逻辑一致；整理版增加失败退出码，便于自动验收。

```
#!/usr/bin/env python3
"""Four-GPU copy correctness check; run in the existing PyTorch container."""
import os
import sys
import torch

torch.set_num_threads(1)
print("PyTorch:", torch.__version__, flush=True)
print("PyTorch CUDA:", torch.version.cuda, flush=True)
print("Visible devices:", os.getenv("CUDA_VISIBLE_DEVICES"), flush=True)
count = torch.cuda.device_count()
print("GPU count:", count, flush=True)
if count != 4:
    raise SystemExit("Expected 4 GPUs; check container visibility/mapping.")
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    uid = getattr(p, "uuid", "unavailable")
    print(f"GPU {i}: {p.name}, UUID={uid}", flush=True)

print("\n=== CUDA peer-access capability ===", flush=True)
for i in range(count):
    row = ["-" if i == j else
           str(int(torch.cuda.can_device_access_peer(i, j)))
           for j in range(count)]
    print(f"GPU {i}: {' '.join(row)}", flush=True)

failures = []

def check(label, actual, expected):
    bad = actual.ne(expected)
    nbad = int(bad.sum().item())
    if nbad:
        indexes = bad.nonzero(as_tuple=True)[0][:5]
        examples = [(int(k), int(expected[k]), int(actual[k]))
                    for k in indexes]
        print(f"FAIL {label}: mismatches={nbad}, "
              f"(index, expected, actual)={examples}", flush=True)
        failures.append(label)
        return False
    return True
```

## 附录 A 四卡传输测试源码续

```
print("\n=== Single-GPU checks ===", flush=True)
reference = (torch.arange(1048576, dtype=torch.int64) * 17 + 13) % 248320
for i in range(count):
    x = reference.to(f"cuda:{i}")
    torch.cuda.synchronize(i)
    copy_ok = check(f"GPU{i} roundtrip", x.cpu(), reference)
    y = x + 7
    torch.cuda.synchronize(i)
    compute_ok = check(f"GPU{i} integer compute", y.cpu(), reference + 7)
    print(f"GPU {i}: roundtrip={'PASS' if copy_ok else 'FAIL'}, "
          f"compute={'PASS' if compute_ok else 'FAIL'}", flush=True)
    del x, y

print("\n=== Directed GPU-to-GPU copy checks ===", flush=True)
for src in range(count):
    for dst in range(count):
        if src == dst:
            continue
        pair_ok = True
        print(f"Testing GPU {src} -> GPU {dst}...", flush=True)
        for n in (32, 4096, 1048576):
            for repeat in range(3):
                expected = (torch.arange(n, dtype=torch.int64) * 17
                            + 13 + repeat * 101) % 248320
                x = expected.to(f"cuda:{src}")
                torch.cuda.synchronize(src)
                y = x.to(f"cuda:{dst}")
                torch.cuda.synchronize(dst)
                label = f"{src}->{dst} direct size={n} repeat={repeat}"
                direct_ok = check(label, y.cpu(), expected)
                host = x.cpu()
                z = host.to(f"cuda:{dst}")
                torch.cuda.synchronize(dst)
                label = f"{src}->{dst} staged size={n} repeat={repeat}"
                staged_ok = check(label, z.cpu(), expected)
                pair_ok = pair_ok and direct_ok and staged_ok
                del x, y, z, host
        print(f"GPU {src} -> GPU {dst}: "
              f"{'PASS' if pair_ok else 'FAIL'}", flush=True)

print("\n=== Summary ===", flush=True)
if failures:
    print(f"FAILED checks: {len(failures)}", flush=True)
    for item in failures:
        print(item, flush=True)
else:
    print("All tested copies and integer checks PASSED.", flush=True)
    print("Basic correctness test, not a bandwidth benchmark.", flush=True)
sys.exit(1 if failures else 0)
```

全量比较在 CPU 上执行，不额外对未经验证的 token ID 做 embedding 索引。测试不包含模型，不依赖 tokenizer，也不依赖 NCCL；它用于确认复制后的整数值是否与源数据完全一致。

## 附录 B Chat 复测与交付文件

```
#!/usr/bin/env python3
"""Optional HF Chat smoke test; may need additional GPU memory."""
import os
from llamafactory.chat import ChatModel

default_path = (
    "/root/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/"
    "snapshots/master"
)
chat = ChatModel({
    "model_name_or_path": os.getenv("MODEL_PATH", default_path),
    "template": "qwen3_5",
    "finetuning_type": "full",
    "infer_backend": "huggingface",
    "max_new_tokens": 512,
    "do_sample": False,
})
model = chat.engine.model
print("Device map:", getattr(model, "hf_device_map", None), flush=True)
devices = sorted({str(p.device) for p in model.parameters()})
print("Parameter devices:", devices, flush=True)
for response in chat.chat([{"role": "user", "content": "你好"}]):
    print(response.response_text, flush=True)
print("Chat completed successfully.", flush=True)
```

max_new_tokens 为 512。对于带思考输出的模型，达到长度上限时可能仍只输出部分思考内容；应结合完成状态和异常判断，不能把截断的思考文本直接判成服务器故障。

| 文件 | 用途 |
| --- | --- |
| L40S_IOMMU_Guide.pdf | 适合阅读与转发的固定版式说明 |
| L40S_IOMMU_Guide.docx | 可编辑的说明文档 |
| L40S_IOMMU_Guide.md | 便于复制命令的文本说明 |
| scripts/collect_system.sh | 只读收集系统、拓扑、ACS、日志和容器版本 |
| scripts/transfer_check.py | 完整的四卡数据正确性测试 |
| scripts/run_transfer.sh | 在现有容器内运行测试并保存日志 |
| scripts/chat_smoke.py | 原模型的命令行 Chat 可用性检查 |

脚本不会自动修改 GRUB、自动重启或自动升级任何软件。完整包中的脚本已做语法检查；服务器实测结论来自用户提供的前后日志及 Chat 成功确认，不代表在编写文档的环境中重新运行了 GPU 测试。

## 附录 C 证据来源与技术参考

### 本机证据

1. llama.log：原 LlamaFactory 模型加载与 CUDA 异常记录。

2. l40s_system_diagnosis.txt：2026-10-22 16:25 左右的修改前系统、拓扑、ACS 与错误日志。

3. l40s_transfer_check.txt：修改前四卡整数复制测试，47 个失败子测试。

4. l40s_system_diagnosis (1).txt：重启后的系统与 ACS 状态。

5. l40s_transfer_check (1).txt：重启后四卡单卡检查及全部 12 个方向通过。

6. 用户提供的 /proc/cmdline、journalctl 与 dmesg 输出：确认 intel_iommu=off 和 DMAR: IOMMU disabled；2026-10-22 用户确认 Chat 恢复。

### 官方参考资料

[1] NVIDIA CUDA Programming Guide
Programming Systems with Multiple GPUs，参见 Host IOMMU Hardware、PCI Access Control Services and VMs。

https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html

[2] NVIDIA NCCL GPU troubleshooting
P2P 状态与实际验证、Linux 裸机 IOMMU、ACS，以及 nvbandwidth 测试建议。

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html

[3] Linux 6.8 内核启动参数
intel_iommu=off 用于关闭 Intel IOMMU 驱动；intremap 是独立参数。

https://docs.kernel.org/6.8/admin-guide/kernel-parameters.html

[4] NVIDIA nvbandwidth
GPU 与系统间带宽测试工具，按仓库说明选择测试项和构建环境。

https://github.com/NVIDIA/nvbandwidth

[5] NVIDIA nccl-tests
NCCL 集合通信的正确性及性能测试工具。

https://github.com/NVIDIA/nccl-tests

本手册限定于记录中的配置与测试范围。网页及 latest 镜像可能更新，长期复现应保存镜像 ID、工具版本和原始日志；不将历史通过结果直接外推到未来的驱动或硬件配置。
