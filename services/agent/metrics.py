from prometheus_client import Counter, Gauge, Histogram

CALLS = Counter("agent_calls_total", "Calls answered by the voice agent")
ACTIVE = Gauge("agent_active_calls", "Calls currently handled by the voice agent")
BARGE_INS = Counter("agent_barge_in_total", "Times the caller interrupted the agent")
TOOLS = Counter("agent_tool_decisions_total", "Routing decisions", ["tool", "source"])
TRANSFERS = Counter("agent_transfers_total", "Warm transfers to a human", ["result"])
STAGE = Histogram(
    "agent_stage_seconds", "Latency per pipeline stage; e2e = caller stops talking -> agent audio starts",
    ["stage"], buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
)
