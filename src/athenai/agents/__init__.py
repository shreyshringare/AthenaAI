"""
Agent runtime package.

HOW IT FITS IN THE SYSTEM:
Exposes the agent loop that the gateway (athenai.gateway.app) drives per
request: Agent (facade) -> AgentExecutor (loop) -> ToolRegistry (tools) and
a model.generate() call, with AgentStatus/AgentStep/AgentResult (state.py)
carrying the trace back to the caller.
"""
