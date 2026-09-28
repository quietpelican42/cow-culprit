"""Runnable demo of both halves of cow-culprit.

    python examples/demo.py

Requires the package installed with its runtime dependencies (pandas, numpy).
"""

from __future__ import annotations

import textwrap

import pandas as pd

from cow_culprit import CowCulpritError, check_source, explain


def demo_linter() -> None:
    print("=== linter ===")
    source = textwrap.dedent(
        """
        import pandas as pd


        def normalize(df: pd.DataFrame) -> None:
            v = df.values
            v -= v.mean(axis=0)
        """
    )
    findings = check_source(source, path="normalize.py")
    for finding in findings:
        print(finding.format())
    print()


def demo_runtime() -> None:
    print("=== runtime explainer ===")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    try:
        with explain():
            v = df.values
            v -= 1
    except CowCulpritError as exc:
        print(f"{type(exc).__name__}: {exc}")
    print()


if __name__ == "__main__":
    demo_linter()
    demo_runtime()
