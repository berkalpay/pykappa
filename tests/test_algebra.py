import pytest
from pykappa.system import System


@pytest.mark.parametrize(
    "expression, result",
    [
        ("[true] [?] 1 [:] 0", 1),
        ("[false] [?] 1 [:] 0", 0),
        ("[max] (1) (4)", 4),
        ("[min](1)(4)", 1),
    ],
)
def test_expression_evaluation(expression, result):
    assert System.from_ka(f"%obs: 'x' {expression}")["x"] == result


@pytest.mark.parametrize(
    "expression, result",
    [
        ("1 + 2 * 3", 7),
        ("1 + 2 * 3 ^ 2", 19),
        ("2 * 2 ^ 3 * 2", 32),
        ("'x' + 2 * 'x'", 30),
    ],
)
def test_operator_precedence(expression, result):
    assert System.from_ka(f"%var: 'x' 10\n%obs: 'y' {expression}")["y"] == result
