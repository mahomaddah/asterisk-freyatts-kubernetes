#!/usr/bin/env bash
# Open the lab UIs on this machine:
#   http://localhost:8091  Call Inspector (voice-agent)
#   http://localhost:3000  Grafana ("Freya Voice Lab" dashboard; password in ~/.freya-lab-grafana.txt)
# kubectl port-forward pins one pod even when given a Service and dies on every rollout,
# so each forward runs in a reconnect loop.
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/freya-lab.yaml}
forward() {  # namespace service local:remote
  while true; do kubectl -n "$1" port-forward "svc/$2" "$3" >/dev/null 2>&1; sleep 2; done
}
forward freya voice-agent 8091:8090 &
forward monitoring monitoring-grafana 3000:80 &
wait
