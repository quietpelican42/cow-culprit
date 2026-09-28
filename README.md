# cow-culprit

Find in-place mutation of pandas-derived arrays that break under Copy-on-Write, and name the pandas object when the error still happens at runtime.

```python
from cow_culprit import explain

with explain():
    v = df.values
    v -= 1
```

```
cow_culprit.runtime.CowCulpritError: output array is read-only

`v` is a read-only view into the DataFrame `df` (pandas Copy-on-Write, mandatory since
pandas 3.0). Fix: obtain the array with copy=True (to_numpy(copy=True)) or .copy(), or
write back through df.loc[...] instead of mutating the array.
```

## Why

pandas 3.0 makes Copy-on-Write mandatory. As a result, `.values`, `.to_numpy()` (without
`copy=True`), `np.asarray(df)` and `.array` can all return a read-only view of pandas'
internal buffer instead of a fresh, mutable array. Code that used to work — extract an
array, mutate it in place — now fails, and the failure gives no hint that pandas is
involved. Measured under pandas 3.0.6 / numpy 2.5.3:

```
>>> v = df.values; v -= 1
ValueError: output array is read-only
>>> np.random.shuffle(s.to_numpy())
ValueError: array is read-only
```

Neither message names pandas, Copy-on-Write, `.values`, or the object the array came
from. Grepping the traceback doesn't help either, because the error is raised inside
numpy, several frames away from the pandas call that produced the array.

Two known trackers describe the same gap. [pandas-dev/pandas#62144](https://github.com/pandas-dev/pandas/issues/62144)
(open) shows the same error for `Series.__setitem__`, `.update()`, and `.replace(inplace=True)`,
and quotes a user report: "I've been encountering this issue when upgrading to Pandas 3.0.
The past behavior when extracting NumPy arrays from Pandas objects is that you get a
mutable NumPy array, but now I'm seeing such errors pop up". pandas' own [Copy-on-Write
migration guide](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) documents
the read-only behavior of `.to_numpy()` but not the augmented-assignment or in-place-numpy-function
idioms that trip over it. The guide's `pd.options.mode.copy_on_write = "warn"` mode,
which people may remember from pandas 2.2, does not help here either — it is deprecated
and a no-op on pandas >= 3.0:

```
Measured: python -c "import pandas as pd; pd.options.mode.copy_on_write = 'warn'"
<string>:1: Pandas4Warning: The 'mode.copy_on_write' option is deprecated. Copy-on-Write
can no longer be disabled (it is always enabled with pandas >= 3.0), and setting the
option has no impact. This option will be removed in pandas 4.0.
```

Incumbents checked: ruff's pandas-vet (`PD011`) flags bare `.values` usage on style
grounds ("ambiguous return type"), but does not look at what happens to the array
afterward, does not cover `.to_numpy()`, `np.asarray(df)` or `.array`, and has no
runtime component. Neither gap is covered end to end by an existing tool.

## Installation

```sh
pip install cow-culprit
```

## Usage

### Lint a codebase for the idiom before it breaks

```sh
cow-culprit path/to/project
```

```
path/to/project/etl.py:14:0: COW001 augmented assignment (Sub) mutates `v` in place, bound from `.values` at line 13 (use copy=True / .copy() when obtaining the array, or write back through df.loc)
```

Exit code `0` means clean, `1` means findings were reported, `2` means a file could
not be parsed.

### Explain the error when it happens anyway

```python
from cow_culprit import explain

with explain():
    risky_function(df)
```

Raises `cow_culprit.CowCulpritError` (a `ValueError` subclass) chained from the
original exception, with the pandas object named when it can be confirmed.

### Or install it globally for a script

```python
import cow_culprit
cow_culprit.install_excepthook()
```

Prints the same diagnosis to stderr ahead of the normal traceback for any
uncaught read-only-buffer `ValueError`, then calls through to whatever
`sys.excepthook` was set before.

## Reference

### CLI

| Argument | Description |
| --- | --- |
| `paths` | One or more `.py` files or directories (searched recursively) to check. |

Exit codes: `0` clean, `1` findings reported, `2` a file could not be parsed / no
Python files found.

### Linter codes

| Code | Flags |
| --- | --- |
| `COW001` | Augmented assignment (`v -= 1`, `v[:] += 1`) on a name bound from a pandas array accessor. |
| `COW002` | Slice/index assignment (`v[i] = x`) on such a name. |
| `COW003` | `out=<name>` keyword argument targeting such a name in any call. |
| `COW004` | An in-place-mutating call on such a name: `np.random.shuffle`, `np.copyto`, `np.put`, `np.place`, or the `.sort()` / `.fill()` / `.resize()` methods. |

### Python API

| Name | Description |
| --- | --- |
| `check_source(source, path="<string>") -> list[Finding]` | Lint source text. |
| `check_file(path) -> list[Finding]` | Lint a file on disk. |
| `Finding` | `path`, `line`, `col`, `code`, `message`, `fix`; `.format()` renders one CLI-style line. |
| `explain()` | Context manager; see above. |
| `CowCulpritError` | The `ValueError` subclass `explain()` raises. |
| `install_excepthook()` / `uninstall_excepthook()` | Global `sys.excepthook` variant. |

## How it works

**Linter.** A single-pass `ast` walk per function body (and one for module-level top
code), tracking which local names are currently bound to a pandas-array-producing
expression: `.values`, `.array` (always), `.to_numpy()` without `copy=True`, or
`np.asarray(x)` where `x` is a name with positive evidence of being pandas (built by
`pd.DataFrame`/`pd.Series`/`pd.array`/`pd.read_*`, a `.copy()` of one of those, or an
annotated parameter). A name's risk is cleared the moment it is reassigned from
something else, including `.copy()`. Every augmented assignment, slice assignment,
`out=` keyword, and known in-place-mutating call against a currently-risky name is
reported with the line where the array was obtained and a fix.

**Runtime explainer.** Pandas makes every Copy-on-Write-managed buffer read-only,
whether or not it is actually shared with another object — so a read-only flag alone
never proves *which* pandas object is responsible. `explain()` walks the raised
exception's traceback frame by frame, collecting every local `numpy.ndarray` that is
read-only and every local `pandas.Series` / `DataFrame` / `Index`, then checks each
pair with `numpy.shares_memory` to confirm which pandas object actually backs the
offending array before naming it.

## Limitations

- **Linter — `.values` / `.array` have no provenance check.** They are always assumed
  to come from a pandas object, per pandas' own documented idiom. This produces false
  positives on unrelated objects that happen to expose a same-named attribute or
  method: measured against pandas' own source (see below), 2 of the linter's 13
  findings were `self.values` / `self.array` on pandas' *internal* `Block` /
  `BlockManager` engine classes, which are not the public `DataFrame.values` /
  `Series.array` this tool targets, and were already guarded by an explicit
  read-only check or a defensive copy immediately before the mutation.
- **Linter — `.to_numpy()` / `np.asarray(x)` don't know about extension-array
  exceptions.** Nullable extension dtypes (`boolean`, `Float64`, Arrow-backed string
  and timedelta arrays) sometimes hand back a writable array from their zero-copy
  `.to_numpy()` / `.array` path instead of a read-only one — the opposite of the
  numpy-backed case this tool is built around. Measured: 5 of the 13 pandas findings
  were exactly this (2 nullable-extension `to_numpy()` tests, 3 Arrow-backed `.array`
  mutations), all confirmed writable by re-running the exact line.
- **Linter — provenance is per function, in program order, and ignores branches** — a
  name marked risky inside one `if` branch is still treated as risky after the `if`
  block even on the path that skipped it. Tuple/list-unpacking assignments and
  attribute-target assignments are not tracked as new bindings. Nested functions and
  lambdas are analyzed independently; a name bound in an enclosing scope is not
  tracked inside a closure.
- **Linter — `np.asarray(x)` requires positive evidence that `x` is pandas**: `x` must
  be a name assigned from `pd.DataFrame(...)` / `pd.Series(...)` / `pd.array(...)` /
  `pd.read_*(...)` / a `.copy()` of one of those, or a parameter annotated
  `DataFrame`/`Series`. An earlier, more permissive version of this rule (flag
  `np.asarray(x)` for any `x` that isn't obviously a literal) was measured against
  scikit-learn and produced 9 findings, all 9 false positives (`np.asarray()` on a
  plain Python list, a PIL image, and a scipy/graph array) — the array-conversion
  idiom scikit-learn uses to accept "anything array-like" is common code far beyond
  pandas, and the earlier rule couldn't tell those apart. This version found 0 in the
  same code, which is correct: scikit-learn's `X`/`graph` parameters here are never
  themselves `pd.DataFrame`/`pd.Series` objects at that call site — a real narrowing,
  not just going quiet. The cost is that an unannotated function whose argument
  happens to be a DataFrame at the call site (`def f(df): np.asarray(df)`, no type
  hint) is not flagged; only `.values` / `.array` catch that case unconditionally.
- **Runtime explainer:** if the read-only array was never bound to a variable — for
  example `np.random.shuffle(s.to_numpy())`, where the temporary array is created and
  handed straight to a C/Cython function — there is no frame anywhere on the stack
  holding a reference to it, so `explain()` cannot confirm which object it came from.
  It falls back to naming the pandas objects visible on the stack as candidates,
  explicitly flagged as unconfirmed, rather than guessing.
- Neither part understands `df.copy(deep=False)`, custom subclasses of `ndarray`, or
  arrays that became read-only for a reason unrelated to pandas (e.g. a memory-mapped
  file); the runtime explainer only claims a pandas connection when it can show shared
  memory with a live pandas object.

### Noise floor

Measured by running the linter over the working tree of 3 real, actively-maintained
repositories that use pandas or numpy array-conversion idioms heavily, and manually
classifying every finding as true or false positive by reading the surrounding code
and, where it was not obvious, running the exact line against real pandas 3.0.6 /
numpy 2.5.3:

| Repo (commit) | Files scanned | Findings | True positive | False positive |
| --- | --- | --- | --- | --- |
| [pandas-dev/pandas](https://github.com/pandas-dev/pandas) @ `7a4b27f` | 1,530 | 13 | 4 | 9 |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) @ `8578499` | 1,037 | 0 | — | — |
| [rasbt/mlxtend](https://github.com/rasbt/mlxtend) @ `93ac19f` | 212 | 0 | — | — |

The 4 true positives are all in pandas' own Copy-on-Write test suite
(`tests/copy_view/`, `tests/indexes/multi/test_missing.py`) — code that deliberately
constructs the exact scenario this tool targets and asserts it raises. The 9 false
positives are the two classes described above (internal engine `.values`/`.array`,
and extension-array zero-copy-but-writable exceptions); both are specific to pandas'
own source, which manages the CoW machinery itself and exercises its exceptions far
more densely than application code that merely uses pandas. scikit-learn and mlxtend,
which consume pandas/numpy rather than implement it, were quiet — 0 findings, 0 false
positives — which is what "low noise on ordinary code" should look like.

## Compatibility

Tested on Python 3.14.7 with pandas 3.0.6 and numpy 2.5.3 (the only Python this
machine has). CI runs the test suite on 3.11, 3.12, 3.13 and 3.14, but that is
"runs in CI", not independently verified here.

## Development

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy src --strict
```

## License

MIT. See [LICENSE](LICENSE).
