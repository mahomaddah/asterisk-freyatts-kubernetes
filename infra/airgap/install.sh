#!/usr/bin/env bash
# Install freya-voice from an offline bundle. Needs k3s (installed from its air-gap tarball),
# the NVIDIA driver + container toolkit, and helm. Nothing here touches the internet.
#
#   sudo ./install.sh            (run inside the bundle directory)
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS                      # the transfer media is untrusted until checked
k3s ctr -n k8s.io images import images.tar   # images land in containerd; pullPolicy IfNotPresent
kubectl create namespace freya --dry-run=client -o yaml | kubectl apply -f -
kubectl -n freya get secret freya-voice-secrets >/dev/null 2>&1 || {
  echo "create the secret first: kubectl -n freya create secret generic freya-voice-secrets --from-env-file=site.env"
  exit 1
}
helm upgrade -i freya-voice ./freya-voice -n freya "$@"
