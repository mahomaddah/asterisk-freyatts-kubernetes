from prometheus_client import Counter, Gauge, Histogram

CALLS = Counter("agent_calls_total", "Calls answered by the voice agent")
ACTIVE = Gauge("agent_active_calls", "Calls currently handled by the voice agent")
BARGE_INS = Counter("agent_barge_in_total", "Times the caller interrupted the agent")
TOOLS = Counter("agent_tool_decisions_total", "Routing decisions", ["tool", "source"])
LLM_ERRORS = Counter("agent_llm_errors_total", "LLM calls that failed or timed out (degraded to a human)")
TRANSFERS = Counter("agent_transfers_total", "Warm transfers to a human", ["result"])
STAGE = Histogram(
    "agent_stage_seconds", "Latency per pipeline stage; e2e = caller stops talking -> agent audio starts",
    ["stage"], buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
)


def init_labels(tools, sources=("guard", "llm", "output_guard", "llm_down", "dtmf", "fallback"),
                transfer_results=("connected", "no_answer")):
    """Create every labelled series at 0 on start-up. Prometheus only sees a labelled counter
    after its first increment and treats that first value as the baseline, so increase()
    silently dropped the first card block, the first transfer, etc. after each restart."""
    for tool in tools:
        for source in sources:
            TOOLS.labels(tool, source)
    for result in transfer_results:
        TRANSFERS.labels(result)
    for stage in ("stt", "llm", "tts", "e2e"):
        STAGE.labels(stage)
