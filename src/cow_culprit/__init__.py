"""cow-culprit: find and explain pandas Copy-on-Write read-only array bugs.

Two independent tools:

* :func:`cow_culprit.linter.check_source` / :func:`check_file` -- a static,
  stdlib-only ``ast`` linter that flags in-place mutation of arrays obtained
  from pandas accessors.
* :class:`cow_culprit.runtime.explain` / :func:`install_excepthook` -- a
  runtime helper that, when a read-only-buffer ``ValueError`` is raised,
  tries to name the pandas object responsible and suggests the fix.
"""

from .linter import Finding, check_file, check_source
from .runtime import CowCulpritError, explain, install_excepthook, uninstall_excepthook

__all__ = [
    "CowCulpritError",
    "Finding",
    "check_file",
    "check_source",
    "explain",
    "install_excepthook",
    "uninstall_excepthook",
]

__version__ = "0.1.0"
