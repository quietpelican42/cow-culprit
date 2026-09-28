import sys

import numpy as np
import pandas as pd
import pytest

from cow_culprit.runtime import CowCulpritError, explain, install_excepthook, uninstall_excepthook


def test_explain_confirms_dataframe_values_augassign() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(CowCulpritError) as excinfo:
        with explain():
            v = df.values
            v -= 1
    message = str(excinfo.value)
    assert "df" in message
    assert "DataFrame" in message
    assert "Copy-on-Write" in message
    assert "copy=True" in message
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_explain_confirms_series_to_numpy_slice_assign() -> None:
    s = pd.Series([1, 2, 3])
    with pytest.raises(CowCulpritError) as excinfo:
        with explain():
            arr = s.to_numpy()
            arr[0] = 0
    message = str(excinfo.value)
    assert "s" in message
    assert "Series" in message


def test_explain_falls_back_when_array_never_bound_to_a_name() -> None:
    # np.random.shuffle(s.to_numpy()) never binds the array to a variable,
    # so explain() cannot confirm which object it came from -- it should
    # still say something useful, naming s as a candidate, without claiming
    # a confirmed match.
    s = pd.Series([1, 2, 3])
    with pytest.raises(CowCulpritError) as excinfo:
        with explain():
            np.random.shuffle(s.to_numpy())
    message = str(excinfo.value)
    assert "Copy-on-Write" in message
    assert "s" in message
    assert "Could not locate" in message


def test_explain_leaves_unrelated_valueerrors_alone() -> None:
    with pytest.raises(ValueError) as excinfo:
        with explain():
            raise ValueError("not related to read-only arrays at all")
    assert not isinstance(excinfo.value, CowCulpritError)


def test_explain_leaves_non_valueerrors_alone() -> None:
    with pytest.raises(TypeError):
        with explain():
            raise TypeError("wrong type")


def test_explain_leaves_read_only_error_alone_with_no_pandas_on_stack() -> None:
    arr = np.zeros(3)
    arr.flags.writeable = False
    with pytest.raises(ValueError) as excinfo:
        with explain():
            arr -= 1
    # No pandas object anywhere on the stack sharing memory or otherwise --
    # explain() should not fabricate a pandas diagnosis for a plain numpy
    # read-only array.
    assert not isinstance(excinfo.value, CowCulpritError)


def test_no_exception_passes_through_untouched() -> None:
    with explain():
        x = 1 + 1
    assert x == 2


def test_install_and_uninstall_excepthook_round_trip() -> None:
    original = sys.excepthook
    install_excepthook()
    assert sys.excepthook is not original
    install_excepthook()  # idempotent
    assert sys.excepthook is not original
    uninstall_excepthook()
    assert sys.excepthook is original


def test_excepthook_writes_diagnosis_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    original = sys.excepthook
    install_excepthook()
    try:
        try:
            v = df.values
            v -= 1
        except ValueError:
            exc_type, exc, tb = sys.exc_info()
            assert exc_type is not None and exc is not None
            sys.excepthook(exc_type, exc, tb)
    finally:
        uninstall_excepthook()
    assert sys.excepthook is original
    captured = capsys.readouterr()
    assert "cow-culprit:" in captured.err
    assert "Copy-on-Write" in captured.err
