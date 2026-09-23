# Asterisk + FreyaTTS on Kubernetes

A home-lab, on-prem **Voice AI call center** for regulated industries (banks, insurance) at mini scale:
Asterisk / SIP telephony → real-time STT → LLM → **FreyaTTS** (Turkish TTS) → caller,
deployed with Helm onto a Kubernetes cluster with a GPU worker node and **zero internet egress**.

**Stack:** Asterisk (PJSIP, ARI, DTMF, warm transfer) · SIP / WebRTC · Kamailio · Kubernetes (k3s) · Helm · Docker ·
NVIDIA device plugin (GPU scheduling) · faster-whisper · llama.cpp (OpenAI-compatible API) · FreyaTTS-small ·
Prometheus / Grafana / Loki · Langfuse · Splunk HEC · Keycloak (OIDC / SAML 2.0) · CyberArk Conjur OSS · Terraform · Python

> Status: work in progress. See `docs/` for the architecture and build log.

## Early benchmark: FreyaTTS-small on Apple M4 (Mac mini, 24 GB)

| Device | short (2.6 s audio) | medium (6.8 s) | long (8.9 s) |
|---|---|---|---|
| CPU fp32 | RTF 0.70 | RTF 0.63 | RTF 0.55 |
| MPS (Metal GPU) | RTF 0.38 | RTF 0.31 | RTF 0.29 |

Reproduce: `python bench/quick_bench.py cpu mps`
