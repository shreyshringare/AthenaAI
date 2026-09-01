"""CalculatorTool — safe arithmetic via AST walk. Never uses eval().

HOW IT FITS IN THE SYSTEM:
Registered into a ToolRegistry (athenai.tools.registry) at gateway startup
(athenai.gateway.app._build_tool_registry) and looked up by name from there
whenever the model emits a "calculator" tool call inside AgentExecutor's loop
(athenai.agents.executor).

WHY AST WALK (NOT eval):
eval() executes arbitrary Python — an attacker passing "__import__('os').system('rm -rf /')"
would run it. AST walk only permits a whitelist of node types: numeric literals,
binary operators, and unary minus. Any other node type raises a ToolDeniedError
before any computation occurs.
"""

from __future__ import annotations

import ast
from typing import Any, ClassVar

from athenai.core.exceptions import ToolDeniedError

_ALLOWED_BINOPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.FloorDiv: "//",
    ast.Mod: "%",
}


def _eval_node(node: ast.expr) -> float | int:
    """Recursively evaluate a single AST node from the whitelisted subset.

    Recurses into BinOp/UnaryOp operands, so the whole expression tree is
    walked node-by-node rather than compiled — any node type not explicitly
    matched (function calls, attribute access, names, comprehensions, ...)
    falls through to the final case and raises ToolDeniedError.
    """
    match node:
        case ast.Constant(value=v) if isinstance(v, int | float):
            return v
        case ast.BinOp(left=left, op=op, right=right):
            op_type = type(op)
            if op_type not in _ALLOWED_BINOPS:
                raise ToolDeniedError(f"unsupported operator: {op_type.__name__}")
            lv = _eval_node(left)
            rv = _eval_node(right)
            match op:
                case ast.Add():
                    return lv + rv
                case ast.Sub():
                    return lv - rv
                case ast.Mult():
                    return lv * rv
                case ast.Div():
                    if rv == 0:
                        raise ToolDeniedError("division by zero")
                    return lv / rv
                case ast.Pow():
                    return lv**rv
                case ast.FloorDiv():
                    if rv == 0:
                        raise ToolDeniedError("division by zero")
                    return lv // rv
                case ast.Mod():
                    if rv == 0:
                        raise ToolDeniedError("modulo by zero")
                    return lv % rv
                case _:
                    raise ToolDeniedError(f"unsupported operator: {type(op).__name__}")
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval_node(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _eval_node(operand)
        case _:
            raise ToolDeniedError(
                f"expression contains disallowed node type: {type(node).__name__}"
            )


class CalculatorTool:
    """Tool protocol implementation exposing safe arithmetic to the agent loop.

    Division, floor-division, and modulo by zero raise ToolDeniedError rather
    than Python's ZeroDivisionError, so the agent executor sees a uniform
    error type it already knows how to surface as a failed tool result.
    """

    name = "calculator"
    description = "Evaluates arithmetic expressions safely using AST walk — no eval()."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate (e.g. '2 + 3 * 4')",
            }
        },
        "required": ["expression"],
    }

    async def execute(self, arguments: dict[str, Any]) -> float | int:
        expression = arguments["expression"]
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ToolDeniedError(f"invalid expression: {exc}") from exc
        return _eval_node(tree.body)
