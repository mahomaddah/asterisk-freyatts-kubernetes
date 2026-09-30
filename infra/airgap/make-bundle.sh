#!/usr/bin/env bash
# Build an offline install bundle for a site with no internet access (the bank case).
# Run on a node that has the images (k3s containerd). Produces:
#   bundle/images.tar      every image the chart needs, exported from containerd
#   bundle/freya-voice/    the Helm chart
#   bundle/install.sh      imports images into k3s, installs the chart
#   bundle/SHA256SUMS      integrity check for the transfer media
#
#   sudo infra/airgap/make-bundle.sh /tmp/freya-bundle
set -euo pipefail
OUT=${1:-/tmp/freya-bundle}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VALUES=$ROOT/deploy/charts/freya-voice/values.yaml
REGISTRY=$(awk '/^  registry:/{print $2; exit}' "$VALUES")

mkdir -p "$OUT"
# image list = the per-component tags pinned in values.yaml, plus the registry itself
mapfile -t IMAGES < <(awk '/^  tags:/{f=1;next} f&&/^    [a-z-]+:/{gsub(/[":]/," ");print $1":"$2} f&&!/^    /{f=0}' "$VALUES" \
                      | sed "s#^#${REGISTRY}/#")
IMAGES+=("docker.io/library/registry:2")
printf '%s\n' "${IMAGES[@]}" > "$OUT/images.txt"

k3s ctr -n k8s.io images export "$OUT/images.tar" "${IMAGES[@]}"
cp -r "$ROOT/deploy/charts/freya-voice" "$OUT/"
cp "$ROOT/deploy/registry/registry.yaml" "$ROOT/infra/k3s/registries.yaml" "$ROOT/infra/airgap/install.sh" "$OUT/"
(cd "$OUT" && sha256sum images.tar > SHA256SUMS)
du -sh "$OUT"/*
