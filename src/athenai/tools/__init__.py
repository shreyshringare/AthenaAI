"""
Tool implementations package.

HOW IT FITS IN THE SYSTEM:
Each module here (calculator.py, http.py, sql.py) implements the Tool
protocol (athenai.core.protocols.Tool) — a concrete capability an agent can
invoke during its reasoning loop. base.py defines the shared ToolResult type
and registry.py provides ToolRegistry, which the gateway (athenai.gateway.app)
populates at startup and AgentExecutor (athenai.agents.executor) queries to
build its system prompt and dispatch tool calls. validator.py provides
ToolValidator for callers that want to check a tool call's arguments and
permissions before invoking execute().
"""
