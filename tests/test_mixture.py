import pytest

from pykappa.pattern import Pattern, Component
from pykappa.mixture import ComponentMixture, neighborhood


def test_neighborhood_requires_frontier_growth_for_radius_gt_1():
    # Chain A - B - C. From A, radius=2 must reach C.
    comp = Component.from_kappa("A(x[1]), B(x[1], y[2]), C(y[2])")
    a = next(agent for agent in comp if agent.type == "A")

    types_r1 = {ag.type for ag in neighborhood([a], radius=1)}
    assert types_r1 == {"A", "B"}

    types_r2 = {ag.type for ag in neighborhood([a], radius=2)}
    assert types_r2 == {"A", "B", "C"}


@pytest.mark.parametrize(
    "test_str",
    [
        "A(a[.]{u})",
        "A(a1[1]), B(b1[1])",
        "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3]), C(c1[3], c2[4], c3[5]), D(d1[4], d2[1])",
        # "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3])",
    ],
)
def test_instantiate_pattern_one_component(test_str):
    pattern = Pattern.from_kappa(test_str)
    mixture = ComponentMixture([pattern])
    assert pattern.components.pop().isomorphic(mixture.components.pop())


@pytest.mark.parametrize(
    "test_case",
    [
        ("A(a1[1]), B(b1[1])", 1, "A(a1[1]), B(b1[1])", 1),
        ("A(a1[1]), A(a1[1])", 1, "A(a1[1]), A(a1[1])", 2),
        ("A(a1[1]), A(a1[1]), A(a1[.]), A(a1[.])", 1, "A(a1[_])", 2),
        ("A(a1[1]), A(a1[1]), A(a1[.]), A(a1[.])", 1, "A(a1[.])", 2),
        ("A(a1[1]), A(a1[1]), A(a1[.]), A(a1[.])", 1, "A(a1[#])", 4),
        (
            "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3]), C(c1[3], c2[4], c3[5]), D(d1[4], d2[1])",
            1,
            "A(a1[1], a2[2], a3[5]), B(b1[2], b2[3]), C(c1[3], c2[4], c3[5]), D(d1[4], d2[1])",
            1,
        ),
    ],
)
def test_find_embeddings_one_component(test_case):
    """
    Test embeddings of patterns consisting of a single component
    """
    n = 1000

    mixture_pattern_str, n_copies, match_pattern_str, n_embeddings_expected = test_case
    mixture_pattern = Pattern.from_kappa(mixture_pattern_str)

    mixture = ComponentMixture()
    for _ in range(n):
        mixture.instantiate(mixture_pattern, n_copies)

    match_pattern = Pattern.from_kappa(match_pattern_str)
    assert len(match_pattern.components) == 1

    embeddings = list(match_pattern.components[0].embeddings(mixture))
    assert len(embeddings) == n_embeddings_expected * n
