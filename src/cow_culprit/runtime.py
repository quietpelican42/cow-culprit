"""Runtime explainer for pandas Copy-on-Write read-only array errors.

pandas >= 3.0 makes Copy-on-Write mandatory, which means arrays obtained
from ``.values``, ``.to_numpy()``, ``np.asarray(df)`` or ``.array`` can be
read-only. Mutating one raises a bare numpy ``ValueError`` such as
``output array is read-only`` or ``array is read-only`` -- the message never
mentions pandas or Copy-on-Write.

This module catches that specific error and tries to name the pandas object
whose buffer was involved, by walking the exception's traceback and looking
for:

1. A read-only :class:`numpy.ndarray` bound to a local variable somewhere on
   the stack, and a pandas ``Series``/``DataFrame``/``Index`` also on the
   stack whose own buffer shares memory with it (checked with
   :func:`numpy.shares_memory`). When found, this is a confirmed match.
2. If no such array is found -- which happens when the array was never
   bound to a name, e.g. ``np.random.shuffle(s.to_numpy())`` -- CPython
   simply has nowhere to keep a reference to it once control passes into a
   C/Cython function such as :func:`numpy.random.shuffle`, so there is
   nothing to inspect. In that case this module falls back to listing the
   pandas objects visible on the stack as *candidates*, without claiming to
   have confirmed which one, because pandas makes every CoW-managed buffer
   read-only regardless of whether it is actually shared -- so "this
   object's buffer is read-only" is true of nearly every pandas object and
   is not, by itself, evidence of *which* one was involved.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from types import FrameType, TracebackType
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "CowCulpritError",
    "explain",
    "install_excepthook",
    "uninstall_excepthook",
]

_READ_ONLY_MARKERS = ("read-only", "read only")

_FIX = (
    "obtain the array with copy=True (to_numpy(copy=True)) or .copy(), "
    "or write back through <object>.loc[...] instead of mutating the array"
)


class CowCulpritError(ValueError):
    """A ValueError caused by mutating a read-only pandas Copy-on-Write buffer.

    Raised in place of the original ``ValueError`` by :class:`explain`,
    chained via ``raise ... from`` so the original traceback is preserved.
    """


def _looks_like_read_only_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and any(marker in str(exc) for marker in _READ_ONLY_MARKERS)


def _iter_frames(tb: TracebackType | None) -> Iterator[FrameType]:
    while tb is not None:
        yield tb.tb_frame
        tb = tb.tb_next


def _pandas_buffer(obj: object) -> np.ndarray | None:
    """Read (never mutate) the ndarray backing a pandas object, or None."""
    try:
        arr: object
        if isinstance(obj, pd.DataFrame):
            arr = obj.values
        elif isinstance(obj, (pd.Series, pd.Index)):
            arr = obj.to_numpy()
        else:
            return None
        return arr if isinstance(arr, np.ndarray) else None
    except Exception:
        return None


def _diagnose(tb: TracebackType | None) -> str | None:
    """Best-effort diagnosis of a read-only-array ValueError. Returns a
    human-readable explanation, or None if nothing useful was found."""
    readonly_arrays: list[tuple[str, np.ndarray]] = []
    pandas_objects: list[tuple[str, Any]] = []

    for frame in _iter_frames(tb):
        for name, value in list(frame.f_locals.items()):
            if isinstance(value, np.ndarray) and not value.flags.writeable:
                readonly_arrays.append((name, value))
            elif isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
                pandas_objects.append((name, value))

    # Path 1: a read-only array is directly reachable -- try to confirm which
    # pandas object it came from via shared memory.
    for arr_name, arr in reversed(readonly_arrays):
        for obj_name, obj in pandas_objects:
            buf = _pandas_buffer(obj)
            if buf is None:
                continue
            try:
                shares = np.shares_memory(arr, buf)
            except Exception:
                shares = False
            if shares:
                kind = type(obj).__name__
                return (
                    f"`{arr_name}` is a read-only view into the {kind} `{obj_name}` "
                    f"(pandas Copy-on-Write, mandatory since pandas 3.0). Fix: {_FIX.replace('<object>', obj_name)}."
                )
        # Read-only array found, but couldn't link it to a specific pandas
        # object on the stack. Only worth reporting if there is at least one
        # pandas object in scope at all -- otherwise this is most likely a
        # read-only array with nothing to do with pandas (e.g. one flagged
        # read-only by hand), and claiming Copy-on-Write would be a guess.
        if pandas_objects:
            return (
                f"`{arr_name}` is read-only. This matches the pandas Copy-on-Write "
                f"read-only-buffer pattern (mandatory since pandas 3.0: .values / "
                f".to_numpy() / np.asarray(df) can return a read-only buffer), but "
                f"no pandas object sharing its memory was found on the stack to "
                f"confirm which one it came from. Fix: {_FIX}."
            )

    # Path 2: no read-only array reachable at all -- this happens when the
    # array was never bound to a name (e.g. np.random.shuffle(s.to_numpy())),
    # so CPython has no local variable to hold a reference once control
    # passes into numpy's C/Cython implementation. Fall back to naming the
    # pandas objects visible on the stack as candidates, without claiming to
    # have confirmed which one -- every CoW-managed buffer is read-only
    # regardless of whether it is actually shared, so that alone isn't proof.
    if pandas_objects:
        names = ", ".join(f"`{name}` ({type(obj).__name__})" for name, obj in pandas_objects)
        return (
            "Could not locate the specific read-only array on the stack -- this "
            "usually means it was a temporary passed directly into a call (e.g. "
            "np.random.shuffle(s.to_numpy())) rather than bound to a variable. "
            "This still matches the pandas Copy-on-Write read-only-buffer pattern "
            f"(mandatory since pandas 3.0). Pandas objects visible on the stack: {names}. "
            f"If one of these is the source, fix it there: {_FIX}."
        )

    return None


class explain:
    """Context manager that enriches pandas Copy-on-Write read-only errors.

    Usage::

        with explain():
            v = df.values
            v -= 1  # raises CowCulpritError naming df, instead of a bare
                     # "ValueError: output array is read-only"

    Only touches ``ValueError``\\ s whose message mentions "read-only" (or
    "read only"); everything else propagates unchanged. If the error looks
    like a read-only-buffer error but no pandas object can be implicated,
    the original exception is left alone rather than replaced with a
    diagnosis that adds nothing.
    """

    def __enter__(self) -> "explain":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if exc is None or not _looks_like_read_only_error(exc):
            return False
        diagnosis = _diagnose(tb)
        if diagnosis is None:
            return False
        raise CowCulpritError(f"{exc}\n\n{diagnosis}") from exc


_previous_hook: Callable[[type[BaseException], BaseException, TracebackType | None], None] | None = None


def _cow_culprit_excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    if _looks_like_read_only_error(exc):
        diagnosis = _diagnose(tb)
        if diagnosis is not None:
            sys.stderr.write(f"cow-culprit: {diagnosis}\n\n")
    hook = _previous_hook or sys.__excepthook__
    hook(exc_type, exc, tb)


def install_excepthook() -> None:
    """Install a sys.excepthook that annotates uncaught read-only-buffer
    ValueErrors with a pandas Copy-on-Write diagnosis, then calls through to
    whatever hook was previously installed."""
    global _previous_hook
    if sys.excepthook is _cow_culprit_excepthook:
        return
    _previous_hook = sys.excepthook
    sys.excepthook = _cow_culprit_excepthook


def uninstall_excepthook() -> None:
    """Restore the excepthook that was active before install_excepthook()."""
    global _previous_hook
    if sys.excepthook is _cow_culprit_excepthook:
        sys.excepthook = _previous_hook or sys.__excepthook__
    _previous_hook = None
