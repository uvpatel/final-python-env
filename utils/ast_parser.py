"""Static parsing helpers for multi-domain Python code analysis."""

from __future__ import annotations

import ast
from typing import Any, Dict, List


class _LoopDepthVisitor(ast.NodeVisitor):
    """Collect loop nesting depth for a parsed Python module."""

    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0

    def _visit_loop(self, node: ast.AST) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self._visit_loop(node)


def parse_code_structure(code: str) -> Dict[str, Any]:
    """Parse Python code into reusable structural signals."""

    summary: Dict[str, Any] = {
        "syntax_valid": True,
        "syntax_error": "",
        "imports": [],
        "function_names": [],
        "class_names": [],
        "loop_count": 0,
        "branch_count": 0,
        "max_loop_depth": 0,
        "line_count": len(code.splitlines()),
        "long_lines": 0,
        "tabs_used": "\t" in code,
        "trailing_whitespace_lines": 0,
        "uses_numpy": False,
        "uses_pandas": False,
        "uses_torch": False,
        "uses_sklearn": False,
        "uses_fastapi": False,
        "uses_flask": False,
        "uses_pydantic": False,
        "uses_recursion": False,
        "calls_eval": False,
        "calls_no_grad": False,
        "calls_backward": False,
        "calls_optimizer_step": False,
        "route_decorators": [],
        "docstring_ratio": 0.0,
        "code_smells": [],
    }

    lines = code.splitlines()
    summary["long_lines"] = sum(1 for line in lines if len(line) > 88)
    summary["trailing_whitespace_lines"] = sum(1 for line in lines if line.rstrip() != line)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        summary["syntax_valid"] = False
        summary["syntax_error"] = f"{exc.msg} (line {exc.lineno})"
        summary["code_smells"].append("Code does not parse.")
        return summary

    visitor = _LoopDepthVisitor()
    visitor.visit(tree)
    summary["max_loop_depth"] = visitor.max_depth

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    summary["function_names"] = [node.name for node in functions]
    summary["class_names"] = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    summary["docstring_ratio"] = (
        sum(1 for node in functions if ast.get_docstring(node)) / len(functions)
        if functions
        else 0.0
    )

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            summary["loop_count"] += 1
        elif isinstance(node, (ast.If, ast.Try, ast.Match)):
            summary["branch_count"] += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "eval":
                summary["calls_eval"] = True
            elif attr == "backward":
                summary["calls_backward"] = True
            elif attr == "step":
                summary["calls_optimizer_step"] = True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            summary["code_smells"].append("Debug print statements are present.")
        elif isinstance(node, ast.With):
            if any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == "no_grad" for item in node.items):
                summary["calls_no_grad"] = True

    import_set = sorted(set(imports))
    summary["imports"] = import_set
    summary["uses_numpy"] = "numpy" in import_set or "np" in code
    summary["uses_pandas"] = "pandas" in import_set or "pd" in code
    summary["uses_torch"] = "torch" in import_set
    summary["uses_sklearn"] = "sklearn" in import_set
    summary["uses_fastapi"] = "fastapi" in import_set
    summary["uses_flask"] = "flask" in import_set
    summary["uses_pydantic"] = "pydantic" in import_set or "BaseModel" in code

    for node in functions:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == node.name:
                summary["uses_recursion"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                    summary["route_decorators"].append(decorator.func.attr)
                elif isinstance(decorator, ast.Attribute):
                    summary["route_decorators"].append(decorator.attr)

    if summary["long_lines"]:
        summary["code_smells"].append("Long lines reduce readability.")
    if summary["tabs_used"]:
        summary["code_smells"].append("Tabs detected; prefer spaces for consistency.")
    if summary["trailing_whitespace_lines"]:
        summary["code_smells"].append("Trailing whitespace found.")

    return summary
