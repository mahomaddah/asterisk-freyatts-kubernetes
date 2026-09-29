#!/usr/bin/env bash
# Bootstrap a single-node k3s cluster on an Ubuntu host with an NVIDIA GPU.
# Tested: Ubuntu 22.04 (HWE kernel 6.8), GTX 1050 Mobile 4 GB, driver 580, k3s v1.36.
#
#   sudo ./install-gpu-node.sh <node-name> <extra-tls-san>
set -euo pipefail
NODE_NAME=${1:-gpu-laptop}
TLS_SAN=${2:-$(hostname).local}

# 1. The NVIDIA kernel module must match the running kernel.
#    Trap we hit: the driver package was held (apt-mark hold) while Ubuntu kept
#    upgrading the kernel; the new kernel's prebuilt module needed a newer 580.x
#    userland, so nothing matched and nvidia-smi failed. Upgrade within the same
#    branch (580 is the last one supporting Pascal), then hold again.
if ! nvidia-smi >/dev/null 2>&1; then
  held=$(apt-mark showhold | grep -x nvidia-driver-580 || true)
  [ -n "$held" ] && apt-mark unhold nvidia-driver-580
  apt-get install -y nvidia-driver-580 "linux-modules-nvidia-580-$(uname -r)"
  [ -n "$held" ] && apt-mark hold nvidia-driver-580
  modprobe nvidia && modprobe nvidia_uvm
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. nvidia-container-toolkit lets containerd start GPU containers; k3s detects it
#    at start-up and registers the "nvidia" RuntimeClass by itself.
dpkg -s nvidia-container-toolkit >/dev/null 2>&1 || {
  echo "install nvidia-container-toolkit first: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  exit 1
}

# 3. k3s server (control plane + worker on this node)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --node-name ${NODE_NAME} \
  --write-kubeconfig-mode 644 --tls-san ${TLS_SAN}" sh -

echo "next (from the workstation): infra/k3s/gpu-device-plugin.sh"
