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

## Correction (November 2026)
The measurements above were made on Apple MPS only. `bench/device_compare.py` repeats them on 17 sentences with the same
scoring at 48 kHz. FreyaTTS seeds a generator on the inference device (`model.py`, `sample()`), and the initial noise is
the voice, so the same seed gives different audio per device:

| device | mean CER | CER > 4 % | prompts with a window < 170 Hz |
|---|---|---|---|
| CUDA (GTX 1050, torch 2.8) | 5.5 % | 4/17 | 1 (`card-info`, 133 Hz) |
| Apple MPS (torch 2.14) | 9.5 % | 9/17 | 2 (95 Hz) |
| CPU (M4, torch 2.14) | 3.3 % | 6/17 | 2 (136 Hz) |

The drift on `card-info` and phrase repetition on `menu` occur on CUDA too, but MPS overstated both. Torch versions differ,
so the noise is the likely cause, not a proven one. An earlier commit said the greeting was clean on MPS and broken on
the GTX 1050; that comparison mixed 8 kHz and 48 kHz audio and is withdrawn. At 8 kHz Whisper does hear an extra syllable in the
CUDA greeting that it does not hear at 48 kHz, so prompt QC should score the 8 kHz version. The call path itself is clean: recorded
audio matches the source with a constant offset and no gaps.

## QC at the rate the caller hears (November 2026)
The paragraph above ends with "prompt QC should score the 8 kHz version". It should, and it had not been: the renderer
scored the 48 kHz render, then resampled to 8 kHz and shipped that unchecked. Transcribing the files Asterisk actually
plays (`services/agent/prompts/*.wav`, 8 kHz PCM) shows what slipped through:

| phrase | CER at 48 kHz | CER at 8 kHz | heard at 8 kHz |
|---|---|---|---|
| `greeting` | 1.8 % | **7.1 %** | "**Fiyaya Bank** hoşgeldiniz ben Leyla…" |
| the other 8 phrases | ≤ 3.6 % | ≤ 3.6 % | correct |

The model loses the one word a bank cannot afford to lose: its own name. This is not the telephone band being worse in
general — corpus CER is the same at both rates (16.2 % vs 17.2 %, above) — it is the `Fr` cluster in *Freya* surviving at
48 kHz and collapsing at 8 kHz.

Two fixes were tried against the 8 kHz transcript:

| attempt | CER 8 kHz | lowest pitch window | usable |
|---|---|---|---|
| as-is, one long clause (`max_words=11`) | 7.1 % | 240 Hz | no, wrong word |
| same text split into short clauses (`max_words=6`) | 0.0 % | **112 Hz** | no, the voice drifts |
| spell it **"Freyya"** for the model, score against "Freya" | 0.0 % (3 of 3 settings) | 202 Hz | yes |

So `render_prompts_verified.py` now (a) transcribes every candidate at 48 kHz **and** at 8 kHz and requires both to pass,
(b) accepts `real text ~ what to feed the model` in the phrase files, so a pronunciation spelling can rescue a word
without changing what the agent claims to say — CER is still scored against the real spelling, and the real spelling is
what lands in `manifest.json` and in the agent's phrase constants, and (c) writes that manifest itself (`--manifest`),
because a key that drifts from `services/agent/call.py` silently turns a verified recording into an unverified live
synthesis. Without the pronunciation spelling the QC winner was "Merhaba, ben Leyla…", which passes every check by
dropping the bank's name altogether: a gate that is allowed to choose the words will optimise the message away.

The renderer is reproducible on one device: re-running it rendered byte-identical audio for every phrase whose winning
candidate did not change. Across devices it is not — that is the per-device generator above.

Takeaway for production: verify at the sample rate, codec and bandwidth the user will hear, not the one the model emits.
A proper noun belongs in a pronunciation lexicon, and a QC gate must judge the words it was given, never pick easier ones.
