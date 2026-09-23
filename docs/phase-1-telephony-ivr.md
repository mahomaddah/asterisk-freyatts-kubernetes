# Phase 1: Asterisk IVR voiced by FreyaTTS

**Goal:** a real phone (iPhone softphone over Wi-Fi) calls a bank line, hears a Turkish
menu spoken by FreyaTTS, navigates with DTMF and can be handed to a human agent.

```
iPhone (Linphone, ext 1001) ──SIP/UDP 5060──► Asterisk 20 (Docker) ──Dial──► Mac Linphone (ext 1002, "agent")
                             ◄──RTP 10000-10099 (G.711 µ-law)──┘
                                                  │ plays 8 kHz prompts
FreyaTTS HTTP service (host, Apple Metal) ──► ASR-verified renderer ──► sounds/tr/freya/*.wav
```

## What was built
| Piece | Notes |
|---|---|
| `telephony/asterisk` | Ubuntu 24.04 + Asterisk 20.6, PJSIP only, explicit module list (`autoload = no`) for a small attack surface and clean logs. Configs are templates rendered by `envsubst` at start-up so secrets live in `.env`, not in the image. |
| NAT handling | Asterisk sits behind Docker NAT: `external_media_address` / `external_signaling_address`, `rtp_symmetric`, `force_rport`, `rewrite_contact`, `direct_media = no`. RTP range limited to 100 ports so it can later be exposed through a Kubernetes Service. |
| Dialplan | `100` → IVR: `1` statement, `2` lost card, `0` transfer to agent `1002`, invalid/timeout handling. DTMF via RFC 4733. |
| `services/tts` | FastAPI wrapper around FreyaTTS-small. Picks `cuda` → `mps` → `cpu`, resamples server-side to the telephony rate, exports `tts_real_time_factor` to Prometheus. |

## Verification
* `channel originate Local/100@from-internal` from the Asterisk CLI: exercises the dialplan without a phone.
* `baresip` registers as 1001, dials 100 and records the received RTP to a WAV: an end-to-end check of SIP registration, NAT and media (20 s, RMS 0.055, non-silent).
* Manual: iPhone call, DTMF 1/2, `0` rang the agent softphone and bridged the call.

## Finding: FreyaTTS-small on real IVR prompts
Whisper (`large-v3-turbo`) round-trip of the first prompt set showed two failure modes:

1. **Phrase repetition / substitution**, e.g. the menu spoke "kayıp veya çalıntı kart için" twice and
   "ödeme tutarı" became "ödeme tarihi" (a copy of an earlier phrase).
2. **Speaker drift inside one utterance.** Pitch tracked per 1.5 s window fell from ~320 Hz to ~95 Hz
   on `card-info`: the voice turns male halfway through. FreyaTTS has no speaker embedding; the initial
   noise is the voice and long inputs are synthesized clause by clause, so some clauses leave the voice.

The 8 kHz telephone band did **not** make intelligibility worse (corpus CER 16.2% at 8 kHz vs 17.2% at 48 kHz);
the errors come from synthesis.

Changing sampler settings (Euler steps 32/64, clause length 11/6/4 words) did not fix drift. Rephrasing did.
`scripts/render_prompts_verified.py` therefore renders each prompt from a list of phrasings × settings and keeps the
first candidate with **CER ≤ 4 %** (spaces ignored, digits spelled out) and **every voiced window ≥ 170 Hz**.
Result: 8/8 prompts pass; report in `bench/results/prompt_qc.json`.

Takeaway for production: pre-rendered prompts can be gated automatically. For live LLM replies the same
checks are too slow to run per turn, so Phase 2 keeps replies short (one clause) and logs pitch/CER
asynchronously for monitoring.
