"""The lint that makes the promise mechanical.

bitforge's claim is "no floats in the inference path." Comments and reviews
drift; this repo pins it with an AST gate that CI runs: every function in
bitforge's FORWARD surface (everything except modules named reference/quant)
must be free of:

  * float literals (0.5, 1e-3 ...)
  * float / np.float64 / np.float32 constructors and dtype mentions
  * true division `/` (produces floats in numpy too — use // or >>)
  * math.exp/sqrt/log/... (float APIs by definition)

Run as a test (test_intmath asserts `lint_forward` reports zero) and as a
pre-commit / CI step. The gate itself is the product: deterministic builds
are the point of the whole repo, and a deployment requirement you cannot
enforce is a comment, not a guarantee.
"""
from __future__ import annotations

import ast
import pathlib
from typing import List

FORBIDDEN_CALLS = {"float", "np.float64", "np.float32", "numpy.float64", "numpy.float32"}
FORBIDDEN_MATH = {"exp", "log", "log1p", "sqrt", "pow", "fsum", "mean"}
FLOAT_ATTRS = {"float64", "float32", "float16", "float_"}


def _func_body(fn: ast.FunctionDef) -> ast.AST:
    return fn


def lint_file(path: pathlib.Path) -> List[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    # exempt: docstrings, and anything under a module that imports as reference
    issues: List[str] = []

    class Check(ast.NodeVisitor):
        def __init__(self):
            self.in_doc = False

        def visit_Constant(self, node):
            if isinstance(node.value, float) and not self.in_doc:
                issues.append(f"{path.name}:{node.lineno}: float literal {node.value!r}")
            self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Div):
                issues.append(f"{path.name}:{node.lineno}: '/' true division (float-producing); use '//' or shift")
            self.generic_visit(node)

        def visit_Call(self, node):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                parts = []
                cur = f
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                name = ".".join(reversed(parts))
            if name and (name in FORBIDDEN_CALLS or name.split(".")[0] + "." + (name.split(".")[-1]) in
                         {f"math.{m}" for m in FORBIDDEN_MATH} or name in {f"math.{m}" for m in FORBIDDEN_MATH}):
                issues.append(f"{path.name}:{node.lineno}: forbidden float call {name}()")
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if node.attr in FLOAT_ATTRS and not self.in_doc:
                issues.append(f"{path.name}:{node.lineno}: float dtype reference {node.attr}")
            self.generic_visit(node)

    # skip docstrings entirely: they may mention floats to explain the contract
    doc_positions = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                doc_positions.add(n.body[0].value.lineno)

    class Check2(Check):
        def visit_Constant(self, node):
            if node.lineno in doc_positions:
                return
            super().visit_Constant(node)

        def visit_Attribute(self, node):
            if node.lineno in doc_positions:
                return
            super().visit_Attribute(node)

    Check2().visit(tree)
    return issues


def lint_forward(pkg_dir: pathlib.Path | None = None) -> List[str]:
    root = pkg_dir or pathlib.Path(__file__).parent
    out: List[str] = []
    for p in sorted(root.glob("*.py")):
        if p.name in {"quant.py", "lint.py"} or p.name.startswith("reference"):
            continue                              # float-legal by design
        out.extend(lint_file(p))
    return out
