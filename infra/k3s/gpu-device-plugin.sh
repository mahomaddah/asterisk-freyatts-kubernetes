#!/usr/bin/env bash
# Expose the GPU to the scheduler as the nvidia.com/gpu resource.
set -euo pipefail
NODE=${1:-gpu-laptop}

# The chart only schedules onto nodes that look like GPU nodes. Without Node Feature
# Discovery nobody sets those labels, so the DaemonSet sits at 0 pods. Label explicitly.
kubectl label node "$NODE" nvidia.com/gpu.present=true --overwrite

helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null
helm repo update >/dev/null
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin --create-namespace \
  --set runtimeClassName=nvidia --wait

kubectl get node "$NODE" -o jsonpath='nvidia.com/gpu allocatable: {.status.allocatable.nvidia\.com/gpu}{"\n"}'
