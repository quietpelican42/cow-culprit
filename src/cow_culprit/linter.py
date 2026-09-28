"""Static linter for pandas Copy-on-Write read-only array bugs.

pandas >= 3.0 makes Copy-on-Write mandatory. ``.values``, ``.to_numpy()``
(without ``copy=True``), ``np.asarray(df_or_series)`` and ``.array`` can all
return a read-only view of pandas' internal buffer. Mutating that view in
place raises ``ValueError`` at runtime -- a message that never mentions
pandas or Copy-on-Write, which makes the failure hard to place. This module
finds those mutations statically, using only :mod:`ast`.

Scope and limits (see README ``Limitations``):

* Provenance is tracked per function body (and separately across module-level
  top code), in program order, ignoring branches -- a name classified risky
  in one ``if`` branch is still considered risky after the ``if`` statement
  even if that branch did not execute.
* Nested functions and lambdas get their own, independent pass; a name bound
  in an enclosing scope is not tracked inside a closure.
* Only simple ``name = expr`` and ``name: ann = expr`` bindings update
  provenance. Tuple/list unpacking, attribute targets and multi-target
  chained assignment to non-Name targets are not tracked as new bindings
  (but subscript targets are still checked for in-place mutation).
* ``np.asarray(x)`` is flagged only when ``x`` is a name with positive
  evidence of being a pandas object: assigned from ``pd.DataFrame(...)``,
  ``pd.Series(...)``, ``pd.array(...)``, a ``pd.read_*(...)`` call, a
  ``.copy()`` of another tracked pandas object, or a function parameter
  annotated ``pd.DataFrame`` / ``pd.Series`` / ``DataFrame`` / ``Series``.
  Measured against real code (see README ``Limitations``), flagging
  ``np.asarray(x)`` for *any* ``x`` that isn't obviously a list/array
  produces far more noise than signal outside of pandas' own source, so
  this needs positive evidence rather than an exclude-list.
  ``.values`` and ``.array`` are always treated as pandas accessors, since
  bare (uncalled) attribute access under those names is a pandas idiom
  (this still produces false positives on non-pandas objects that happen
  to expose a same-named attribute; see Limitations).
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterator
from pathlib import Path

#: Dotted call targets whose first positional argument they mutate in place.
_MUTATING_FIRST_ARG_CALLS = {
    "numpy.random.shuffle": "np.random.shuffle(arr) shuffles arr in place",
    "numpy.copyto": "np.copyto(dst, src) writes into dst in place",
    "numpy.put": "np.put(arr, ...) writes into arr in place",
    "numpy.place": "np.place(arr, ...) writes into arr in place",
}

#: Method names on the array itself that mutate it in place.
_MUTATING_METHODS = {"sort", "fill", "resize"}

#: Dotted calls that construct a pandas object -- positive evidence for the
#: name they're assigned to.
_PANDAS_CONSTRUCTOR_CALLS = {
    "pandas.DataFrame",
    "pandas.Series",
    "pandas.array",
}

#: Annotation names that mark a function parameter as a pandas object.
_PANDAS_ANNOTATIONS = {"DataFrame", "Series"}


@dataclasses.dataclass(frozen=True)
class Finding:
    """One flagged location."""

    path: str
    line: int
    col: int
    code: str
    message: str
    fix: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message} ({self.fix})"


@dataclasses.dataclass
class _RiskyBinding:
    """Why a name is considered a possibly read-only pandas-owned buffer."""

    origin: str  # human-readable description, e.g. "`.values` at line 3"


class _Aliases:
    """Resolves import aliases so `np`, `numpy`, and `from numpy import x`
    all map back to canonical dotted names."""

    def __init__(self) -> None:
        # alias -> canonical module path, e.g. {"np": "numpy"}
        self.modules: dict[str, str] = {}
        # bare name -> canonical dotted target, e.g. {"asarray": "numpy.asarray"}
        self.bare: dict[str, str] = {}

    def visit_import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            self.modules[local] = alias.name

    def visit_import_from(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        if node.module not in ("numpy", "numpy.random"):
            return
        for alias in node.names:
            local = alias.asname or alias.name
            canonical = f"{node.module}.{alias.name}"
            self.bare[local] = canonical

    def resolve_call_path(self, func: ast.expr) -> str | None:
        """Return a canonical dotted path for a Call's func, e.g.
        'numpy.random.shuffle' or 'pandas.DataFrame', resolving import
        aliases at the root. None if it can't be resolved (the root isn't a
        plain Name, e.g. `self.foo()`)."""
        parts: list[str] = []
        node: ast.expr = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            if not parts:
                # Bare name call: only resolvable via an explicit
                # `from numpy import x` (tracked) or a handful of builtins.
                if node.id in self.bare:
                    return self.bare[node.id]
                if node.id in {"list", "dict", "set", "tuple", "range"} and node.id not in self.modules:
                    return node.id
                return None
            root = self.modules.get(node.id, node.id)
            return ".".join([root, *reversed(parts)])
        return None


def _is_pandas_constructor_call(expr: ast.expr, aliases: _Aliases) -> bool:
    """True if `expr` is a call that is known to construct/return a pandas
    object: pd.DataFrame(...), pd.Series(...), pd.array(...), pd.read_*(...)."""
    if not isinstance(expr, ast.Call):
        return False
    path = aliases.resolve_call_path(expr.func)
    if path is None:
        return False
    if path in _PANDAS_CONSTRUCTOR_CALLS:
        return True
    return path.startswith("pandas.read_")


def _annotation_is_pandas(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    name: str | None = None
    if isinstance(annotation, ast.Name):
        name = annotation.id
    elif isinstance(annotation, ast.Attribute):
        name = annotation.attr
    elif isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        name = annotation.value.rsplit(".", 1)[-1]
    return name in _PANDAS_ANNOTATIONS


def _describe_source(expr: ast.expr, aliases: _Aliases, lineno: int, pandas_objs: set[str]) -> str | None:
    """If `expr` is a pandas-array-producing expression, return a short
    description of it. Otherwise return None (not tracked as risky)."""
    if isinstance(expr, ast.Attribute) and expr.attr in ("values", "array"):
        return f"`.{expr.attr}` at line {lineno}"

    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == "to_numpy":
            for kw in expr.keywords:
                if kw.arg == "copy" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return None  # explicit copy=True: not risky
            return f"`.to_numpy()` at line {lineno} (no copy=True)"

        path = aliases.resolve_call_path(expr.func)
        if path == "numpy.asarray" and expr.args:
            arg = expr.args[0]
            is_pandas_arg = (isinstance(arg, ast.Name) and arg.id in pandas_objs) or _is_pandas_constructor_call(
                arg, aliases
            )
            if is_pandas_arg:
                return f"`np.asarray(...)` at line {lineno}"
            return None

    return None


def _is_copy_call(expr: ast.expr) -> bool:
    """True if `expr` is `<anything>.copy(...)` -- e.g. `df.values.copy()`
    or `df.copy()`. Any such result is a fresh array/object, not a tracked
    read-only pandas buffer."""
    return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "copy"


def _copy_source_name(expr: ast.expr) -> str | None:
    """If `expr` is `<name>.copy(...)` (base is a plain Name), return that
    name -- used to propagate pandas-object status through a `.copy()`."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "copy":
        if isinstance(expr.func.value, ast.Name):
            return expr.func.value.id
    return None


def _iter_body_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield statements in program order, recursing into control-flow blocks
    (if/for/while/with/try) but NOT into nested function/class scopes."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _iter_body_statements(stmt.body)
            yield from _iter_body_statements(stmt.orelse)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            yield from _iter_body_statements(stmt.body)
            yield from _iter_body_statements(stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield from _iter_body_statements(stmt.body)
        elif isinstance(stmt, ast.Try):
            yield from _iter_body_statements(stmt.body)
            for handler in stmt.handlers:
                yield from _iter_body_statements(handler.body)
            yield from _iter_body_statements(stmt.orelse)
            yield from _iter_body_statements(stmt.finalbody)


class _ScopeChecker:
    """Runs one provenance pass over a single function body or module top
    level, emitting findings in program order."""

    def __init__(self, path: str, aliases: _Aliases, pandas_params: set[str] | None = None) -> None:
        self.path = path
        self.aliases = aliases
        self.risky: dict[str, _RiskyBinding] = {}
        self.pandas_objs: set[str] = set(pandas_params or ())
        self.findings: list[Finding] = []

    def _finding(self, node: ast.AST, code: str, message: str, fix: str) -> None:
        self.findings.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                code=code,
                message=message,
                fix=fix,
            )
        )

    def _bind(self, target: ast.expr, value: ast.expr) -> None:
        if not isinstance(target, ast.Name):
            return  # tuple/attribute/subscript targets: provenance not tracked
        name = target.id

        # Track pandas-object provenance (used only to gate np.asarray(x)).
        copy_source = _copy_source_name(value)
        if _is_pandas_constructor_call(value, self.aliases) or (
            copy_source is not None and copy_source in self.pandas_objs
        ):
            self.pandas_objs.add(name)
        else:
            self.pandas_objs.discard(name)

        if _is_copy_call(value):
            self.risky.pop(name, None)
            return
        origin = _describe_source(value, self.aliases, getattr(value, "lineno", 0), self.pandas_objs)
        if origin is not None:
            self.risky[name] = _RiskyBinding(origin=origin)
        else:
            self.risky.pop(name, None)

    def _check_mutating_call(self, call: ast.Call) -> None:
        # out=<risky name> keyword on any call.
        for kw in call.keywords:
            if kw.arg == "out" and isinstance(kw.value, ast.Name) and kw.value.id in self.risky:
                binding = self.risky[kw.value.id]
                self._finding(
                    call,
                    "COW003",
                    f"`out={kw.value.id}` writes into an array bound from {binding.origin}",
                    "pass a fresh array as out=, or write back through df.loc / copy=True",
                )

        path = self.aliases.resolve_call_path(call.func)
        if path in _MUTATING_FIRST_ARG_CALLS and call.args:
            arg = call.args[0]
            if isinstance(arg, ast.Name) and arg.id in self.risky:
                binding = self.risky[arg.id]
                self._finding(
                    call,
                    "COW004",
                    f"{_MUTATING_FIRST_ARG_CALLS[path]}, and `{arg.id}` is bound from {binding.origin}",
                    "use copy=True / .copy() when obtaining the array, or write back through df.loc",
                )

        if isinstance(call.func, ast.Attribute) and call.func.attr in _MUTATING_METHODS:
            base = call.func.value
            if isinstance(base, ast.Name) and base.id in self.risky:
                binding = self.risky[base.id]
                self._finding(
                    call,
                    "COW004",
                    f"`.{call.func.attr}()` mutates in place, and `{base.id}` is bound from {binding.origin}",
                    "use copy=True / .copy() when obtaining the array, or write back through df.loc",
                )

    def _walk_expr_for_calls(self, expr: ast.expr) -> None:
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                self._check_mutating_call(node)

    def visit_statement(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    if target.value.id in self.risky:
                        binding = self.risky[target.value.id]
                        self._finding(
                            stmt,
                            "COW002",
                            f"slice/index assignment mutates `{target.value.id}`, bound from {binding.origin}",
                            "write back through df.loc[...] = ..., or copy=True / .copy() when obtaining the array",
                        )
            self._walk_expr_for_calls(stmt.value)
            for target in stmt.targets:
                self._bind(target, stmt.value)

        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._walk_expr_for_calls(stmt.value)
                self._bind(stmt.target, stmt.value)

        elif isinstance(stmt, ast.AugAssign):
            target_name: str | None = None
            if isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
            elif isinstance(stmt.target, ast.Subscript) and isinstance(stmt.target.value, ast.Name):
                target_name = stmt.target.value.id
            if target_name is not None and target_name in self.risky:
                binding = self.risky[target_name]
                op = type(stmt.op).__name__
                self._finding(
                    stmt,
                    "COW001",
                    f"augmented assignment ({op}) mutates `{target_name}` in place, bound from {binding.origin}",
                    "use copy=True / .copy() when obtaining the array, or write back through df.loc",
                )
            self._walk_expr_for_calls(stmt.value)

        elif isinstance(stmt, ast.Expr):
            self._walk_expr_for_calls(stmt.value)

        else:
            # Any other statement kind: still scan its expression subtrees
            # for mutating calls (e.g. a Return or a call inside a with-item),
            # but don't recurse into nested statement bodies here -- that is
            # handled by _iter_body_statements at the caller.
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    self._walk_expr_for_calls(child)


def _pre_scan_imports(tree: ast.Module) -> _Aliases:
    aliases = _Aliases()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.visit_import(node)
        elif isinstance(node, ast.ImportFrom):
            aliases.visit_import_from(node)
    return aliases


def check_source(source: str, path: str = "<string>") -> list[Finding]:
    """Analyze one module's source text and return all findings, sorted by
    location."""
    tree = ast.parse(source, filename=path)
    aliases = _pre_scan_imports(tree)
    findings: list[Finding] = []

    # Module-level top code (not inside any function) is its own scope.
    top_level_checker = _ScopeChecker(path, aliases)
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in _iter_body_statements([stmt]):
            top_level_checker.visit_statement(sub)
    findings.extend(top_level_checker.findings)

    # Each function (including nested ones, and methods inside classes) gets
    # its own independent pass. Parameters annotated as DataFrame/Series seed
    # the pandas-object set so `np.asarray(that_param)` can be recognized.
    for func in _walk_all_functions(tree):
        pandas_params = {
            arg.arg
            for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)
            if _annotation_is_pandas(arg.annotation)
        }
        checker = _ScopeChecker(path, aliases, pandas_params)
        for sub in _iter_body_statements(func.body):
            checker.visit_statement(sub)
        findings.extend(checker.findings)

    findings.sort(key=lambda f: (f.line, f.col, f.code))
    return findings


def _walk_all_functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def check_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    return check_source(source, path=str(path))
