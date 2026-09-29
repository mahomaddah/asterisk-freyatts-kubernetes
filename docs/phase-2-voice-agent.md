# Phase 2: Real-time voice agent over ARI

**Goal:** call extension `200`, talk to "Leyla" in Turkish, get real actions done (block a card, read the
statement, reach a human) with barge-in, and see every stage of every turn live.

```
iPhone ──SIP/RTP──► Asterisk ──ARI (REST + WebSocket)──► voice-agent (Python, asyncio)
                     │  mixing bridge                    │
                     └──ExternalMedia RTP (G.711 µ-law)──┤  Silero VAD  →  Whisper STT
                                                         │  guard / LLM router  →  tool (code)
                                                         │  FreyaTTS  →  RTP back to the bridge
                                                         └─ Call Inspector (live timeline, audio)
```

## Call flow
1. Dialplan `200 → Stasis(voice-agent)`. The agent answers, creates a mixing bridge, and adds an
   **ExternalMedia** channel whose RTP is sent to a UDP port the agent opened for that call.
2. **VAD:** Silero on 32 ms windows. Speech starts after 4 voiced windows (a 300 ms pre-roll keeps the
   first syllable); the turn ends after ~600 ms of silence.
3. **Barge-in:** speech while the agent is talking clears the outbound RTP queue and cancels the
   in-flight turn (STT/LLM/TTS), so a stale answer never plays over the caller.
4. **STT → router → tool → TTS**, then RTP is paced back at 20 ms per packet.
5. **Warm transfer:** the agent rings `PJSIP/1002`; when the human answers they first hear a
   spoken briefing (the caller's actual request), then join the caller's bridge and the AI leaves.
   The caller hears a hold message and Turkish ringback (450 Hz, 2 s on / 4 s off) meanwhile.

## Routing: function calling that a 2B model can do reliably
Tools are registered Semantic-Kernel style (`@bank.tool(...)` in `bank_tools.py`); the registry renders
them either as native function-calling schemas or as one **JSON-schema-constrained** choice.

| Qwen3.5 on Apple M4, 8 banking utterances | 2B | 4B |
|---|---|---|
| free-form tool calls | 5/8, ~620 ms | 4/8, ~1.4 s; **said "I blocked your card" without calling the tool** |
| constrained JSON intent | 7/8, ~700 ms | 8/8, ~1.6 s |
| final router (guards + constrained 2B), 14 cases | **14/14, ~850 ms** | |

Design rules that came out of this:
* **The model chooses, code acts.** Sentences spoken after an action come from the tool, never from the
  model, so the model cannot claim an action that did not happen.
* **Deterministic guards where a bank needs them:** "I want a human" (tolerant of STT typos such as
  *temsiciyle*) always transfers; yes/no after "shall I connect you?" is handled in code.
* **Output guard:** free-text replies containing digits or words like *saat, faiz, ücret* are replaced by a
  fixed offer to connect a human. The 2B model invented opening hours despite being told not to.

## Latency (end of caller speech → first agent audio)
| change | e2e |
|---|---|
| first working version (cold LLM, live TTS of a two-sentence confirmation) | 6.8 s |
| LLM kept warm (`keep_alive`) | 3.9 s |
| fixed phrases pre-rendered at start-up | **1.8 s** |

Breakdown of a typical turn: endpointing 608 ms, STT (mlx-whisper small) ~330 ms, router ~850 ms, TTS 0 ms (cache hit).

## Bug story: the AI line was silent
**Symptom:** the Inspector showed every stage working, but the caller heard nothing on `200`; `100` (plain IVR) was fine.

**Investigation:** the test caller's recording had RMS 0. `rtp set debug on` during a test call, packets counted per address:

| direction | packets |
|---|---|
| Asterisk ← caller (PT 0, µ-law) | 394 |
| Asterisk ← agent (PT 118, slin16) | 218 |
| Asterisk → agent | 394 |
| **Asterisk → caller** | **0** |

The agent's audio reached Asterisk and the channel formats/transcoding paths looked correct
(`slin16 → slin → ulaw`). Turning `strictrtp` off changed nothing, which ruled out source-address
learning behind Docker NAT. Remaining suspect: the **dynamic payload type**. ExternalMedia in the
Ubuntu Asterisk 20.6 build sent slin16 as PT 118 but did not map inbound PT 118 back to frames.

**Fix:** ExternalMedia with **G.711 µ-law (static PT 0)**. The phone leg is 8 kHz G.711 anyway, so this also
removed a pointless 8→16→8 kHz transcode. Python 3.13 removed `audioop`, so `g711.py` implements
µ-law in numpy (round-trip SNR 36.7 dB, as expected for µ-law).

**Lesson:** verify media, not logs. The automated test had passed on logs while the audio path was
broken. The smoke test now checks the RMS of what the caller receives.

## Other failures found by testing on a real phone
* Transfer to an agent phone that was offline (`Allocation failed`) left the call in "transferring" with
  VAD paused: the caller was no longer heard. Now: "all agents are busy", keep listening.
  Reproduced on purpose with `HUMAN_AGENT_ENDPOINT=PJSIP/9999`.
* The human was briefed with the caller's last word ("Evet.") instead of the request.
* Replies like "you can talk to an agent" left callers unsure what to say.

## Privacy (KVKK)
* Card, ID and phone numbers are masked in logs (`mask_pii`).
* The Inspector binds to `127.0.0.1` and deletes caller audio after 30 minutes.
* Next: replace the regex with Freya's own `freyavoice/pii-ner-model`.
