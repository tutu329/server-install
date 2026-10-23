#!/usr/bin/env bash
# Read-only collection. Continue collecting if an optional command fails.
set -uo pipefail
container=${1:-llamafactory-webui}
echo '===== System / virtualization ====='
date -Is
uname -a
systemd-detect-virt || true
lscpu
echo '===== Mainboard / BIOS ====='
sudo dmidecode -t baseboard -t bios
echo '===== GPU status / PCIe links ====='
nvidia-smi
fields='index,name,pci.bus_id,driver_version'
fields+=',pcie.link.gen.current,pcie.link.gen.max'
fields+=',pcie.link.width.current,pcie.link.width.max'
nvidia-smi --query-gpu="$fields" --format=csv
echo '===== GPU topology / P2P capability ====='
nvidia-smi topo -m
for capability in r w p; do
    nvidia-smi topo -p2p "$capability"
done
echo '===== Boot command line ====='
cat /proc/cmdline
echo '===== Current-boot kernel diagnostics ====='
sudo journalctl -k -b --no-pager | \
    grep -Ei 'DMAR|IOMMU|AMD-Vi|NVRM|Xid|AER|PCIe Bus Error|DMA.*fault' || true
echo '===== PCIe tree ====='
lspci -Dtv
echo '===== PCIe links / ACS / status ====='
sudo lspci -Dvv | \
    grep -E '^[0-9a-fA-F]{4}:|LnkCap:|LnkSta:|DevCtl:|ACSCtl:|UESta:|CESta:'
echo '===== Container GPU environment ====='
sudo docker exec -i "$container" python - <<'PY'
import os
for k, v in sorted(os.environ.items()):
    if k.startswith(("CUDA_", "NVIDIA_", "NCCL_")):
        print(f"{k}={v}")
PY
echo '===== Installed packages ====='
sudo docker exec -i "$container" python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
for package in ("torch", "transformers", "accelerate", "llamafactory"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: package metadata unavailable")
PY
echo '===== Container image ID ====='
sudo docker inspect --format '{{.Image}}' "$container"
