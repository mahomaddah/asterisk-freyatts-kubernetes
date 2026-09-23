"""Tool registry, in the spirit of Semantic Kernel plugins.

Functions are registered with @registry.tool(...) (like [KernelFunction]); the
registry renders them either as native function-calling schemas (large models)
or as a single constrained-JSON schema where the model *must* pick one tool
(small models, see router.py). Either way the agent executes the tool; the
model never claims an action happened on its own.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

NONE_TOOL = "none"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., dict]
    parameters: dict = field(default_factory=dict)  # JSON-schema "properties"
    required: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def tool(self, description: str, parameters: dict | None = None, required: list[str] | None = None):
        def register(fn):
            self.tools[fn.__name__] = Tool(fn.__name__, description, fn, parameters or {}, required or [])
            return fn
        return register

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict:
        tool = self.tools[name]
        allowed = {k: v for k, v in arguments.items() if k in tool.parameters}
        return tool.fn(**allowed)

    def function_schemas(self) -> list[dict]:
        """OpenAI / Ollama native tool-calling format."""
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {"type": "object", "properties": t.parameters, "required": t.required},
            },
        } for t in self.tools.values()]

    def choice_schema(self) -> dict:
        """JSON schema that forces the model to choose exactly one tool (or none)."""
        merged_args = {}
        for t in self.tools.values():
            merged_args.update(t.parameters)
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [*self.tools, NONE_TOOL]},
                "arguments": {"type": "object", "properties": merged_args},
                "reply": {"type": "string"},
            },
            "required": ["tool", "reply"],
        }

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
