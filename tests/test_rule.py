import pytest
from math import comb

from pykappa.algebra import Expression
from pykappa.pattern import Pattern
from pykappa.rule import KappaRule, KappaRuleUnimolecular, KappaRuleBimolecular
from pykappa.system import System


@pytest.mark.parametrize(
    "test_case",
    [
        ("A(), B()", 10, KappaRule, "B(), A()", 100),
        ("A(), B()", 10, KappaRuleUnimolecular, "B(), A()", 0),
        ("A(), B()", 10, KappaRuleBimolecular, "B(), A()", 100),
        ("A()", 10, KappaRule, "A(), A()", 100),  # No automorphism checks currently
        ("A(a[1]), B(b[1]), C()", 10, KappaRule, "A(a[1]), B(b[1]), C()", 100),
        ("A(a[1]), B(b[1]), C()", 10, KappaRule, "A(), B(), C()", 1000),
        ("A(a[1]), B(b[1])", 10, KappaRuleUnimolecular, "B(), A()", 10),
        ("A(a[1]), B(b[1])", 10, KappaRuleBimolecular, "B(), A()", 90),
        (
            "A(a1[1]), B(b1[1], b2[2]), B(b1[2], b2[3]) A(a2[3])",
            10,
            KappaRuleUnimolecular,
            "B(), A()",
            40,
        ),
    ],
)
def test_rule_n_embeddings_at_system_initialiation(test_case):
    (mixture_pattern_str, n_copies, rule_class, rule_pattern_str, n_embeddings) = (
        test_case
    )
    rule_pattern = Pattern.from_kappa(rule_pattern_str)
    rule = rule_class(rule_pattern, rule_pattern, Expression.from_kappa("1.0"))
    system = System.from_kappa({mixture_pattern_str: n_copies}, [rule.kappa_str])
    assert system.rules["r0"].n_embeddings(system.mixture) == n_embeddings


def test_simple_rule_application():
    """Test selection/application of a simple KappaRule in a mixture."""
    n_copies = 10
    rule_left_str = "A(a[1]), B(b[1])"
    rule = KappaRule(
        Pattern.from_kappa(rule_left_str),
        Pattern.from_kappa("A(a[.]), B(b[.])"),
        Expression.from_kappa("1.0"),
    )
    observables = [f"|{rule_left_str}|", "|A(a[.])|", "|B(b[#])|"]
    system = System.from_kappa(
        {"A(a[1]), B(b[1])": n_copies}, [rule.kappa_str], observables
    )
    rule = system.rules["r0"]

    assert rule.n_embeddings(system.mixture) == n_copies
    assert system["o0"] == n_copies
    assert system["o1"] == 0

    for i in range(1, n_copies + 1):
        update = rule.select(system.mixture)
        assert len(update.edges_to_remove) == 1

        system.mixture.apply_update(update)
        assert system["o0"] == n_copies - i
        assert system["o1"] == i
        assert system["o2"] == n_copies


def test_edge_creating_rule_application():
    """Test selection/application of a KappaRule which creates a new edge in a mixture."""
    n_copies = 4
    rule_right_str = "A(a[1]), B(b[1])"
    rule = KappaRule(
        Pattern.from_kappa("A(a[.]), B(b[.])"),
        Pattern.from_kappa(rule_right_str),
        Expression.from_kappa("1.0"),
    )
    observables = [f"|{rule_right_str}|"]
    system = System.from_kappa(
        {"A(a[.]), B(b[.])": n_copies}, [rule.kappa_str], observables
    )
    rule = system.rules["r0"]

    assert rule.n_embeddings(system.mixture) == n_copies * n_copies
    assert system["o0"] == 0
    for _ in range(1, n_copies + 1):
        update = rule.select(system.mixture)
        assert len(update.edges_to_add) == 1
        system.mixture.apply_update(update)
    assert system["o0"] == n_copies


def test_rule_application():
    """
    Test selection/application of a slightly more involved KappaRule in a mixture.

    TODO: Support for empty slots (".") in pattern strings when instantiating from Lark.
    TODO: Supporting agent interfaces so we know a master list of sites for agents to be able to initialize defaults.
    """
    n_copies = 100
    rule_left_str = "A(a[1]), B(b[1], x[3]), C(c[2]{p}), D(d[2]{p}, x[3])"
    rule = KappaRule(
        Pattern.from_kappa(rule_left_str),
        Pattern.from_kappa("A(a[1]), B(b[.], x[3]), C(c[1]{u}), D(d[.]{p}, x[3])"),
        Expression.from_kappa("1.0"),
    )
    observables = [f"|{rule_left_str}|", "|A(a[1]), C(c[1])|", "|B(b[_])|", "|C(c{u})|"]
    system = System.from_kappa(
        {"A(a[1]), B(b[1], x[3]), C(c[2]{p}), D(d[2]{p}, x[3])": n_copies},
        [rule.kappa_str],
        observables,
    )
    rule = system.rules["r0"]

    assert rule.n_embeddings(system.mixture) == n_copies
    assert system["o0"] == n_copies
    assert system["o1"] == 0

    for i in range(1, n_copies + 1):
        update = rule.select(system.mixture)
        assert len(update.edges_to_remove) == 2
        assert len(update.edges_to_add) == 1
        assert len(update.agents_changed) == 1

        system.mixture.apply_update(update)
        assert system["o0"] == n_copies - i
        assert system["o1"] == i
        assert system["o2"] == n_copies - i
        assert system["o3"] == i


@pytest.mark.parametrize("n_copies", [50])
def test_simple_unimolecular_rule_application(n_copies):
    """Test selection/application of a simple unimolecular KappaRule in a mixture."""
    system = System.from_kappa(
        {"A(a[1]{u}), B(b[1]{u})": n_copies},
        [
            "A(a{u}), B(b{u}) -> A(a{p}), B(b{p}) @ 0.0 {1.0}",
            "A(a[1]), B(b[1]) -> A(a[.]), B(b[.]) @ 1.0",
        ],
        ["|A(a[1]{u}), B(b[1]{u})|"],
    )

    rule1 = system.rules["r0"]
    rule2 = system.rules["r1"]
    assert isinstance(rule1, KappaRuleUnimolecular)
    assert isinstance(rule2, KappaRule)

    assert system["o0"] == n_copies
    n_rule1_applications = n_copies // 2
    n_rule2_applications = n_copies // 2

    for i in range(1, n_rule2_applications + 1):
        update = rule2.select(system.mixture)
        system.mixture.apply_update(update)
        assert system["o0"] == n_copies - i
        assert len(system.mixture.components) == n_copies + i
        assert rule1.n_embeddings(system.mixture) == n_copies - i

    for i in range(1, n_rule1_applications + 1):
        rule1.n_embeddings(
            system.mixture
        )  # Uni/bimolecular rules use this to weight choices
        update = rule1.select(system.mixture)
        system.mixture.apply_update(update)
        assert rule1.n_embeddings(system.mixture) == n_copies - n_rule2_applications - i
        assert system["o0"] == n_copies - n_rule2_applications - i


@pytest.mark.parametrize("n_copies", [50])
def test_simple_bimolecular_rule_application(n_copies):
    """Test selection/application of a simple bimolecular KappaRule in a mixture."""
    system = System.from_kappa(
        {"A(a[.]{u})": n_copies},
        ["A(a{u}), A(a{u}) -> A(a{p}), B(a{p}) @ 1.0 {0.0}"],
        ["|B(a{p})|"],
    )

    rule1 = system.rules["r0"]
    assert isinstance(rule1, KappaRuleBimolecular)

    n_rule1_applications = n_copies // 2
    for i in range(1, n_rule1_applications + 1):
        rule1.n_embeddings(
            system.mixture
        )  # Uni/bimolecular rules use this to weight choices
        update = rule1.select(system.mixture)
        system.mixture.apply_update(update)
        assert rule1.n_embeddings(system.mixture) == 2 * comb(n_copies - 2 * i, 2)
        assert system["o0"] == i


@pytest.mark.parametrize(
    "rule_str, n_symmetries_expected",
    [
        ("A(a[1], b{u}), A(a[1], b{u}) -> A(), A()", 2),
        ("A(a[1], b{u}), A(a[1], b{u}) -> A(b{u}), A(b{p})", 1),
        ("A(x[.]), B(x[.]) -> A(x[1]), B(x[1])", 1),
        (". -> A()", 1),
        ("A() -> .", 1),
        ("., . -> A(), B()", 1),
        ("., . -> A(), A()", 2),
        ("A(), ., . -> A(), B(), C()", 1),
    ],
)
def test_rule_symmetries(rule_str, n_symmetries_expected):
    rule = KappaRule.from_kappa(f"{rule_str} @ 1.0")
    assert rule.n_symmetries == n_symmetries_expected
