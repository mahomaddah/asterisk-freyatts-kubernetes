#!/usr/bin/env bash
# Open the in-cluster Call Inspector on http://localhost:8091.
# kubectl port-forward pins one pod even when given a Service, so it dies on every rollout;
# this loop reconnects to the new pod automatically.
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/freya-lab.yaml}
PORT=${1:-8091}
while true; do
  kubectl -n freya port-forward svc/voice-agent "$PORT:8090" >/dev/null 2>&1
  sleep 2
done
