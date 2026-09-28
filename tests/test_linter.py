from pathlib import Path

import pytest

from cow_culprit.linter import check_file, check_source

FIXTURES = Path(__file__).parent / "fixtures"


def _codes(path: Path) -> list[str]:
    return [f.code for f in check_file(path)]


@pytest.mark.parametrize(
    "fixture,expected_codes",
    [
        ("positive_values_augassign.py", ["COW001"]),
        ("positive_to_numpy_slice_assign.py", ["COW002"]),
        ("positive_asarray_shuffle.py", ["COW004"]),
        ("positive_array_attr_out_kwarg.py", ["COW003"]),
        ("positive_sort_fill_resize.py", ["COW004", "COW004", "COW004"]),
    ],
)
def test_positive_fixtures_are_flagged(fixture: str, expected_codes: list[str]) -> None:
    codes = _codes(FIXTURES / fixture)
    assert codes == expected_codes


@pytest.mark.parametrize(
    "fixture",
    [
        "negative_copy_true.py",
        "negative_explicit_copy_method.py",
        "negative_write_via_loc.py",
        "negative_reassigned_unrelated.py",
    ],
)
def test_negative_fixtures_are_quiet(fixture: str) -> None:
    findings = check_file(FIXTURES / fixture)
    assert findings == [], f"unexpected findings: {[f.format() for f in findings]}"


def test_finding_format_includes_location_and_fix() -> None:
    findings = check_file(FIXTURES / "positive_values_augassign.py")
    assert len(findings) == 1
    line = findings[0].format()
    assert "positive_values_augassign.py:6:4" in line
    assert "COW001" in line
    assert "df.values" in line or ".values" in line
    assert "copy=True" in line


def test_module_level_top_code_is_its_own_scope() -> None:
    source = """
import pandas as pd

df = pd.DataFrame({"a": [1.0]})
v = df.values
v -= 1
"""
    findings = check_source(source, path="mod.py")
    assert [f.code for f in findings] == ["COW001"]


def test_nested_function_gets_independent_scope() -> None:
    source = """
import pandas as pd


def outer(df: pd.DataFrame):
    v = df.values

    def inner():
        # v is not tracked here: closures aren't followed across scopes
        pass

    v -= 1
"""
    findings = check_source(source, path="mod.py")
    assert [f.code for f in findings] == ["COW001"]


def test_to_numpy_with_variable_copy_kwarg_is_conservatively_flagged() -> None:
    # copy=<non-literal> is not proof of a real copy, so we stay conservative.
    source = """
import pandas as pd


def mutate(df: pd.DataFrame, make_copy: bool):
    v = df.to_numpy(copy=make_copy)
    v -= 1
"""
    findings = check_source(source, path="mod.py")
    assert [f.code for f in findings] == ["COW001"]


def test_unparseable_file_raises_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    with pytest.raises(SyntaxError):
        check_file(bad)
