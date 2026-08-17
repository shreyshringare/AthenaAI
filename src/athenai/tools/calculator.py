"""CalculatorTool — safe arithmetic via AST walk. Never uses eval().

WHY AST WALK (NOT eval):
eval() executes arbitrary Python — an attacker passing "__import__('os').system('rm -rf /')"
would run it. AST walk only permits a whitelist of node types: numeric literals,
binary operators, and unary minus. Any other node type raises ValueError before
any computation occurs.
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
