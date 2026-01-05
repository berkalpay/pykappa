import pytest
import shutil
import itertools
import random
from collections import defaultdict

from pykappa.rule import AVOGADRO, DIFFUSION_RATE, kinetic_to_stochastic_on_rate
from pykappa.system import System


def heterodimerization_system(k_on: float = 2.5e9) -> System:
    random.seed(42)
    avogadro = 6.0221413e23
    volume = 2.25e-12  # mammalian cell volume
    n_a, n_b = 1000, 1000
    return System.from_kappa(
        {"A(x[.])": n_a, "B(x[.])": n_b},
        rules=[
            f"A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ {k_on / (avogadro * volume)}",
            "A(x[1]), B(x[1]) -> A(x[.]), B(x[.]) @ 2.5",
        ],
        observables=[f"|A(x[1]),B(x[1])|"],
    )


def test_basic_system():
    system = System.from_kappa(
        {"A(a[.], b[.])": 100},
        rules=[
            "A(a[.]), A(a[.]) <-> A(a[1]), A(a[1]) @ 1.0 {2.0}, 1.0",
            "A(b[.]), A(b[.]) <-> A(b[1]), A(b[1]) @ 1.5, 1.0",
        ],
        observables=[
            "|A(a[.])|",
            "|A(b[1]), A(b[1])|",
            "|A(a[1], b[.]), A(a[1], b[_])|",
        ],
    )

    counts = defaultdict(list)
    for _ in range(1000):
        system.update()
        for obs_name in system.observables:
            counts[obs_name].append(system[obs_name])


def test_basic_observable_symmetry():
    system = System.from_ka(
        """
        %init: 1 V(v[1]), V(v[1])
        %init: 100 V(v[.])
        
        %obs: 'dimer' |V(v[1]), V(v[1])|
        %obs: 'total' 2 * 'dimer' + |V(v[.])|
        """
    )
    assert system["dimer"] == 1
    assert system["total"] == 102


def test_system_from_kappa():
    system = System.from_ka(
        """
    %def: "maxConsecutiveClash" "20"
    %def: "seed" "365457"

    // constants
    %var: 'x'     0.03
    %var: 'k_on'  'x' * 10
    %var: 'g_on'  'k_on' / 100

    %var: 'n' 3 * 100

    %init: 'n' A(a[1]{p}), B(b[1]{u})

    %obs: 'A_total'   |A()|
    %obs: 'A_u'       |A(a{u})|
    %obs: 'B_u'       |B(b{u})|
    %obs: 'A_p'       |A(a{p})|
    %obs: 'pairs'     |A(a[1]), B(b[1])|

    A(a{p}), B(b[_]) -> A(a{u}), B() @ 'g_on'
    """
    )
    n = system["n"]
    assert n == 300
    assert system["g_on"] == 0.003
    assert system["A_total"] == n

    for i in range(1, n):
        system.update()
        assert system["A_total"] == n
        assert system["A_u"] == i
        assert system["B_u"] == n
        assert system["A_p"] == n - i
        assert system["pairs"] == n


@pytest.mark.parametrize(
    "k_on, expected, via_kasim",
    [
        (k_on, expected, vk)
        for vk in ((False, True) if shutil.which("KaSim") else (False,))
        for k_on, expected in [(2.5e8, 65), (2.0e9, 331)]
    ],
)
def test_heterodimerization(k_on, expected, via_kasim):
    system = heterodimerization_system(k_on)
    n_heterodimers = []
    while system.time < 2:
        if via_kasim:
            system.update_via_kasim(0.1)
        else:
            system.update()
        if system.time > 1:
            n_heterodimers.append(system["o0"])

    measured = sum(n_heterodimers) / len(n_heterodimers)
    assert abs(measured - expected) < expected / 5


@pytest.mark.parametrize(
    "kd, a_init, b_init",
    itertools.product([10**-9], [2000], [2000, 3500]),
)
def test_equilibrium_matches_kd(kd, a_init, b_init):
    """
    Check that the input Kd matches what's observed empirically post-equilibrium
    within a relative margin of error.
    """
    volume = 10**-13
    on_rate = kinetic_to_stochastic_on_rate(volume=volume)
    kd = 10**-9
    off_rate = DIFFUSION_RATE * kd
    system = System.from_ka(
        f"""
        %init: {a_init} A(x[.])
        %init: {b_init} B(x[.])
        %obs: 'A' |A(x[.])|
        %obs: 'B' |B(x[.])|
        %obs: 'AB' |B(x[_])|
        A(x[.]), B(x[.]) <-> A(x[1]), B(x[1]) @ {on_rate}, {off_rate}
        """
    )

    empirical_kds = []
    while system.time < 2:
        system.update()
        a_conc_eq = system["A"] / AVOGADRO / volume
        b_conc_eq = system["B"] / AVOGADRO / volume
        ab_conc_eq = system["AB"] / AVOGADRO / volume
        empirical_kds.append(a_conc_eq * b_conc_eq / ab_conc_eq)
    i = int(len(empirical_kds) * 0.5)  # an index post-equilibrium
    empirical_kd = sum(empirical_kds[i:]) / len(empirical_kds[i:])
    assert abs((empirical_kd - kd) / kd) < 0.1


def test_system_manipulation():
    system = System.from_ka(
        """
        %init: 10 A(x[.])
        %init: 10 B(x[.])
        %init: 1 C()

        %obs: 'A' |A(x[.])|
        %obs: 'B' |B(x[.])|
        %obs: 'AB' |A(x[1]), B(x[1])|

        %var: 'total_agents' 'A' + 'B' + (2 * 'AB')

        A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ 1 {1}
        """
    )
    assert not system["AB"]
    system.update()
    assert system["AB"] == 1

    # Add a pattern
    system.mixture.instantiate("A(x[1]), B(x[1])")
    assert system["AB"] == 2
    total_agents_pre_removal = system["total_agents"]
    assert total_agents_pre_removal == 22

    # Remove a component
    component_to_remove = system.mixture.components[0]
    system.mixture.remove(component_to_remove)
    assert total_agents_pre_removal - system["total_agents"] == len(component_to_remove)

    # Add the component back
    system.mixture.add(component_to_remove)
    assert system["total_agents"] == total_agents_pre_removal

    # Set a new variable
    system["twiceA"] = "2 * 'A'"
    assert system["twiceA"] == 2 * system["A"]
    system.update()
    assert system["twiceA"] == 2 * system["A"]

    # Update an observable
    system["A"] = "-1"
    assert system["twiceA"] == -2

    # Set a new observable
    system["C"] = "|C()|"
    assert system["C"] == 1
    system.update()
    assert system["C"] == 1

    system.add_rule("B() -> C() @ 1000", name="new")
    while system.reactivity:
        system.update()
    assert system["B"] == 0 and system["C"] == 12
    system.remove_rule("new")
    system.add_rule("C() -> B() @ 1000")
    while system.reactivity:
        assert system["B"] == 12 and system["C"] == 0


def test_reproducibility_from_initialization():
    """Test that the same system specification can be used to generate reprodicible behavior."""

    seed = 42
    kwargs = {
        "mixture": {"A(x[.])": 100, "B(x[.])": 100},
        "rules": [
            "A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ 0.1",
            "A(x[1]), B(x[1]) -> A(x[.]), B(x[.]) @ 1.0",
        ],
        "observables": {"A_free": "|A(x[.])|"},
        "seed": seed,
    }
    system1 = System.from_kappa(**kwargs)
    system2 = System.from_kappa(**kwargs)
    ka_str = """
        %init: 100 A(x[.])
        %init: 100 B(x[.])

        %obs: 'A_free' |A(x[.])|

        A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ 0.1
        A(x[1]), B(x[1]) -> A(x[.]), B(x[.]) @ 1.0
        """
    system3 = System.from_ka(ka_str, seed=seed)
    system_different = System.from_ka(ka_str, seed=1)

    diverged = False
    for _ in range(100):
        # Systems with same seed should match
        assert system1.time == system2.time == system3.time
        assert system1["A_free"] == system2["A_free"] == system3["A_free"]

        # Check if system with different seed has diverged
        if (
            system1.time != system_different.time
            or system1["A_free"] != system_different["A_free"]
        ):
            diverged = True

        system1.update()
        random.seed(0)  # Make sure external random operations are irrelevant
        random.random()
        system2.update()
        system3.update()
        system_different.update()

    assert (
        diverged
    ), "Systems with different seeds should produce different trajectories"
