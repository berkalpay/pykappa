import pytest
import math
from pathlib import Path

from pykappa.parsing import kappa_parser
from pykappa.pattern import Pattern
from pykappa.rule import KappaRule, KappaRuleUnimolecular, KappaRuleBimolecular
from pykappa.system import System

# Parser


def test_parse_file_without_error():
    kappa_parser.parse_file(
        str(Path(__file__).parent / "wnt_v8.ka")
    )  # 121 rules, 10 agents


@pytest.mark.parametrize(
    "test_kappa", ["A(s[.]), S(a[.]) -> A(s[1]), S(a[1])    @	1", "A(s[.])"]
)
def test_parse_without_error(test_kappa):
    kappa_parser.parse(test_kappa)


# Patterns


def test_pattern_from_kappa():
    test_kappa = """
        A(a[.]{blah}, b[_]{bleh}, c[#], d[some_site_name.some_agent_name], e[13]),
        B(f[13], e[1], z[3]),
        C(x[1]),
        D(w[3]),
        E()
    """
    pattern = Pattern.from_kappa(test_kappa)
    assert ["A", "B", "C", "D", "E"] == [agent.type for agent in pattern.agents]
    assert ["a", "b", "c", "d", "e"] == [site.label for site in pattern.agents[0]]
    assert len(pattern.components) == 2


# Rules


@pytest.mark.parametrize(
    "rule_str,rule_len",
    [
        ("A(a{p}), B(), . -> A(a{u}), B(), C() @ 1.0", 1),
        ("A(a{p}), B(), . <-> A(a{u}), B(), C() @ 1.0, 2.0", 2),
        ("A(b[.]), A(b[.]) <-> A(b[1]), A(b[1]) @ 100.0, 1.0", 2),
    ],
)
def test_rule_len_from_kappa(rule_str, rule_len):
    assert len(KappaRule.list_from_kappa(rule_str)) == rule_len


def test_ambi_rule_from_kappa():
    rules = KappaRule.list_from_kappa(
        "A(a{p}), B(b[1]), C(c[1]) -> A(a{u}), B(b[.]), C(c[.]) @ 1.0 {2.0}"
    )
    assert len(rules) == 2
    assert isinstance(rules[0], KappaRuleBimolecular)
    assert isinstance(rules[1], KappaRuleUnimolecular)
    assert rules[0].stochastic_rate.evaluate() == 1.0
    assert rules[1].stochastic_rate.evaluate() == 2.0


def test_uni_rule_from_kappa():
    rules = KappaRule.list_from_kappa(
        "A(a{p}), B(b[1]), C(c[1]) -> A(a{u}), B(b[.]), C(c[.]) @ 0.0 {2.0}"
    )
    assert len(rules) == 1
    assert isinstance(rules[0], KappaRuleUnimolecular)
    assert rules[0].stochastic_rate.evaluate() == 2.0


def test_bi_rule_from_kappa():
    rules = KappaRule.list_from_kappa(
        "A(a{p}), B(b[1]), C(c[1]) -> A(a{u}), B(b[.]), C(c[.]) @ 1.0 {0.0}"
    )
    assert len(rules) == 1
    assert isinstance(rules[0], KappaRuleBimolecular)
    assert rules[0].stochastic_rate.evaluate() == 1.0


def test_ambi_fr_rule_from_kappa():
    rules = KappaRule.list_from_kappa(
        "A(a{p}), B(b[1]), C(c[1]) <-> A(a{u}), B(b[.]), C(c[.]) @ 1.0 {2.0}, 3.0"
    )
    assert len(rules) == 3
    assert isinstance(rules[0], KappaRuleBimolecular)
    assert isinstance(rules[1], KappaRuleUnimolecular)
    assert rules[0].stochastic_rate.evaluate() == 1.0
    assert rules[1].stochastic_rate.evaluate() == 2.0
    assert rules[2].stochastic_rate.evaluate() == 3.0


# System


def test_system_kappa_str():
    system_in = System.from_ka(
        """
    %var: 'x'     0.03
    %var: 'k_on'  'x' * 10
    %var: 'g_on'  'k_on' / 100

    %var: 'n' 3 * 100
    %var: 'p' [pi] * 'n'
    %var: 'sqpi' [sqrt] ([pi])
    %var: 'm' [max] ('n') (5 * 3)

    %init: 'n' A(a[1]{p}), B(b[1]{u})

    %obs: 'pairs'   |A(a[1]), B(b[1])|

    A(a{p}), B(b[_]) -> A(a{u}), B() @ 'g_on'
        """
    )
    system_out = System.from_ka(system_in.kappa_str)
    for system in (system_in, system_out):
        assert system["pairs"] == system["n"] == system["m"] == 300
        assert system["p"] == pytest.approx(math.pi * system["n"])
        assert system["sqpi"] == pytest.approx(math.sqrt(math.pi))
    assert system_in["g_on"] == system_out["g_on"]
