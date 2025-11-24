import pytest
from pykappa.pattern import Component, Pattern


@pytest.mark.parametrize(
    "test_case",
    [
        ("A(a[.]{u})", "A(a{u}[.])", True),
        ("A(a[.]{u})", "A(a{p}[.])", False),
        ("A(a[#]{#})", "A(a{#}[#])", True),
        (
            "A(a[#]{#})",
            "A(a[.]{#})",
            False,
        ),  # Important distinction vs. embeddings/subgraph isomorphism
        ("A(a[#]{#})", "A(a[#]{u})", False),
        ("A()", "A(a{u})", False),
        ("A(a[.]{u})", "A(a[.])", False),
        ("A(a[1]{u}), A(a[1])", "A(a[1]{u}), B(a[1])", False),
        (
            "A(a1[1]{u}, a2[3]), B(b1[1], b2[2]), C(c1[2], c2[3])",
            "A(a1[1]{u}, a2[3]), C(c1[2], c2[3]), B(b1[1], b2[2])",
            True,
        ),
        (
            "A(a1[1]{u}), B(b1[1], b2[2]), C(c1[2])",
            "A(a1[1]{u}, a2[3]), B(b1[1], b2[2]), C(c1[2], c2[3])",
            False,
        ),
        (
            "A(a1[1]{u}), B(b1[1], b2[2]), C(c1[2])",
            "A(a1[1]{u}, a2[3]), B(b1[1], b2[2]), C(c1[2], c2[3])",
            False,
        ),
        (
            "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3]), C(c1[3], c2[4], c3[5]), D(d1[4], d2[1])",
            "B(b1[2], b2[3]), D(d1[4], d2[1]), C(c1[3], c2[4], c3[5]), A(a1[1], a2[2], a3[5])",
            True,
        ),
        (
            "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3], b3[6]), C(c1[3], c2[4], c3[5]), D(d1[4], d2[1], d3[6])",
            "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3], b3[5]), C(c1[3], c2[4], c3[6]), D(d1[4], d2[1], d3[6])",
            False,
        ),
    ],
)
def test_component_isomorphism(test_case):
    a_str, b_str, isomorphic = test_case
    a = Component.from_kappa(a_str)
    b = Component.from_kappa(b_str)
    assert a.isomorphic(b) == b.isomorphic(a)
    assert isomorphic == a.isomorphic(b)


def test_component_id_uniqueness():
    a = Component.from_kappa("A(a[.]{u})")
    b = Component.from_kappa("A(a[.]{u})")
    assert a.id != b.id


@pytest.mark.parametrize(
    "test_case",
    [
        ("A(a1[1]), A(a1[1])", 2),
        ("A(a1[1]), A(a2[1])", 1),
        ("A(a1[3], a2[1]), A(a1[1], a2[2]), A(a1[2], a2[3])", 3),
    ],
)
def test_automorphism_counting(test_case):
    kappa_str, n_automorphisms_expected = test_case
    component = Component.from_kappa(kappa_str)
    assert component.isomorphic(component)
    assert n_automorphisms_expected == len(list(component.isomorphisms(component)))


def test_pattern_creation():
    Pattern.from_kappa(
        """
        A(a[.]{blah}, b[_]{bleh}, c[#], d[some_site_name.some_agent_name], e[13]),
        B(f[13], e[1], z[3]),
        C(x[1]),
        D(w[3]),
        E()
        """
    )
