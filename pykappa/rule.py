"""Represents Kappa rules."""

import random
from dataclasses import dataclass, field
from math import prod
from typing import Literal, Optional, Self, TYPE_CHECKING
from functools import cached_property
from copy import deepcopy

from pykappa.pattern import Pattern, Component, Agent, Site
from pykappa.mixture import Mixture, _MixtureUpdate
from pykappa.expression import Expression
from pykappa._utils import rejection_sample

if TYPE_CHECKING:
    from pykappa.system import System


@dataclass
class _DifferentComponentTotals:
    """Aggregate embedding counts for a different-component constraint."""

    first: int = 0
    second: int = 0
    overlap: int = 0

    @property
    def weight(self) -> int:
        return self.first * self.second - self.overlap


@dataclass(frozen=True, eq=False)
class Rule:
    """A Kappa rule, specifying the transformation of a pattern at a stochastic rate."""

    left: Pattern
    right: Pattern
    rate_expression: Expression
    component_constraint: Literal["any", "same", "different"] = "any"
    token_updates: tuple[tuple[Expression, str], ...] = ()
    _component_weights: dict[Component, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _component_counts: dict[Component, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _different_totals: _DifferentComponentTotals = field(
        default_factory=_DifferentComponentTotals, init=False, repr=False, compare=False
    )
    _same_weight: int = field(default=0, init=False, repr=False, compare=False)

    @classmethod
    def list_from_kappa(cls, kappa_str: str) -> list[Self]:
        """Parse Kappa string into a list of rules.

        Note:
            Forward-reverse rules (with "<->") represent two rules.
        """
        from pykappa._parsing import kappa_parser, KappaTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        rule_tree = input_tree.children[0]
        return KappaTransformer().transform(rule_tree)

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a single Kappa rule from string.

        Raises:
            AssertionError: If the string represents more than one rule.
        """
        rules = cls.list_from_kappa(kappa_str)
        assert (
            len(rules) == 1
        ), "The given rule expression represents more than one rule."
        return rules[0]

    def __post_init__(self):
        object.__setattr__(
            self,
            "token_updates",
            tuple(
                (expression, name) for expression, name in (self.token_updates or ())
            ),
        )
        l = len(self.left.agents)
        r = len(self.right.agents)
        assert (
            l == r
        ), f"The left-hand side of this rule has {l} slots, but the right-hand side has {r}."
        assert self.component_constraint in {"any", "same", "different"}
        assert (
            self.component_constraint != "different" or len(self.left.components) == 2
        ), "A different-component constraint requires exactly 2 pattern components."

    def __len__(self):
        return len(self.left.agents)

    def __iter__(self):
        yield from zip(self.left.agents, self.right.agents)

    def __repr__(self):
        return f'{type(self).__name__}(kappa_str="{self.kappa_str}")'

    def __str__(self):
        return self.kappa_str

    @property
    def _rate_str(self) -> str:
        rate = self.rate_expression.kappa_str
        if self.component_constraint == "same":
            return f"0 {{{rate}}}"
        if self.component_constraint == "different":
            return f"{rate} {{0}}"
        return rate

    @cached_property
    def kappa_str(self) -> str:
        token_part = ""
        if self.token_updates:
            updates_str = " ".join(
                f"{expr.kappa_str} {name}" for expr, name in self.token_updates
            )
            token_part = f" | {updates_str}"
        return f"{self.left.kappa_str} -> {self.right.kappa_str}{token_part} @ {self._rate_str}"

    @cached_property
    def n_symmetries(self) -> int:
        """
        The number of distinct automorphisms of the graph containing both left- and
        right-hand side agents, augmented with edges between positionally corresponding agents.
        For example, if a rule looks like "l1(...), l2(...) -> r1(...), r2(...)",
        this method draws artifical edges between l1 and r1, and between l2 and r2,
        then returns the number of symmetries of the resulting graph by counting
        how many ways it can be mapped onto itself.
        """
        left_agents = deepcopy(self.left.agents)
        right_agents = deepcopy(self.right.agents)

        for l, r in zip(left_agents, right_agents):
            if l is not None and r is not None:
                l_site = Site("__temp__", "left", partner=None)
                r_site = Site("__temp__", "right", partner=None)

                l_site._set_partner(r_site)
                l._add_site(l_site)

                r_site._set_partner(l_site)
                r._add_site(r_site)

        pattern = Pattern(left_agents + right_agents)
        return pattern.n_isomorphisms(pattern)

    def reactivity(self, system: "System") -> float:
        """Calculate the total reactivity of this rule in the given system,
        i.e. the number of embeddings times the reaction rate, accounting
        for rule symmetry.
        """
        return self._reactivity_from_embeddings(
            self.n_embeddings(system.mixture), system
        )

    def _reactivity_from_embeddings(self, n_embeddings: int, system: "System") -> float:
        """Calculate reactivity without evaluating rates for impossible rules."""
        if not n_embeddings:
            return 0.0
        return n_embeddings // self.n_symmetries * self.rate(system)

    def rate(self, system: "System") -> float:
        return self.rate_expression.evaluate(system)

    def n_embeddings(self, mixture: Mixture) -> int:
        """Count embeddings in the mixture.

        Note:
            This doesn't do any symmetry correction, though `System`
            applies this correction when calculating rule reactivities.
        """
        if self.component_constraint == "same":
            self._component_weights.clear()
            self._component_weights.update(
                (component, prod(self._counts_in_component(mixture, component)))
                for component in mixture.components
            )
            total = sum(self._component_weights.values())
            object.__setattr__(self, "_same_weight", total)
            return total

        if self.component_constraint == "different":
            self._component_counts.clear()
            totals = self._different_totals
            totals.first = totals.second = totals.overlap = 0
            for component in mixture.components:
                counts = self._counts_in_component(mixture, component)
                self._component_counts[component] = counts
                self._add_different_counts(counts)
            return totals.weight

        return prod(
            len(mixture.embeddings(component)) for component in self.left.components
        )

    def _counts_in_component(
        self, mixture: Mixture, component: Component
    ) -> tuple[int, ...]:
        return tuple(
            len(mixture.embeddings_in_component(pattern, component))
            for pattern in self.left.components
        )

    def _add_different_counts(self, counts: tuple[int, ...], sign: int = 1) -> None:
        first, second = counts
        totals = self._different_totals
        totals.first += sign * first
        totals.second += sign * second
        totals.overlap += sign * first * second

    def update_component_weights(
        self,
        mixture: Mixture,
        previous_components: set[Component],
        current_components: set[Component],
    ) -> int:
        """Refresh constraint weights after an event touched some components."""
        if self.component_constraint == "same":
            total = self._same_weight
            for component in previous_components:
                total -= self._component_weights.pop(component, 0)
            for component in current_components:
                weight = prod(self._counts_in_component(mixture, component))
                self._component_weights[component] = weight
                total += weight
            object.__setattr__(self, "_same_weight", total)
            return total

        if self.component_constraint == "different":
            for component in previous_components:
                if counts := self._component_counts.pop(component, None):
                    self._add_different_counts(counts, -1)
            for component in current_components:
                counts = self._counts_in_component(mixture, component)
                self._component_counts[component] = counts
                self._add_different_counts(counts)
            return self._different_totals.weight

    def _select(
        self, mixture: Mixture, rng: random.Random | None = None
    ) -> Optional[_MixtureUpdate]:
        """Select agents and specify the update (or None for invalid match).

        Note:
            Can change the internal states of agents in the mixture but
            records everything else in the MixtureUpdate.
        """
        rng = random if rng is None else rng

        if self.component_constraint != "any":
            components = list(mixture.components)
            if self.component_constraint == "different":
                second_total = self._different_totals.second
                weights = [
                    first * (second_total - second)
                    for first, second in (
                        self._component_counts[component] for component in components
                    )
                ]
            else:
                weights = [
                    self._component_weights[component] for component in components
                ]
            selected_component = rng.choices(
                components,
                weights,
            )[0]

            if self.component_constraint == "different":
                first, second = self.left.components
                return self._produce_update(
                    dict(
                        rng.choice(
                            mixture.embeddings_in_component(first, selected_component)
                        )
                    )
                    | dict(
                        rejection_sample(
                            mixture.embeddings(second),
                            mixture.embeddings_in_component(second, selected_component),
                            rng=rng,
                        )
                    ),
                    mixture,
                )

            embeddings = lambda component: mixture.embeddings_in_component(
                component, selected_component
            )
        else:
            embeddings = mixture.embeddings

        rule_embedding: dict[Agent, Agent] = {}

        for component in self.left.components:
            component_embeddings = (
                embeddings(component)
                if component in mixture._embeddings
                else list(component.embeddings(mixture))
            )
            if not component_embeddings:
                return None
            component_embedding = rng.choice(component_embeddings)

            for rule_agent in component_embedding:
                mixture_agent = component_embedding[rule_agent]
                if mixture_agent in rule_embedding.values():
                    return None  # Invalid match: two selected components intersect
                else:
                    rule_embedding[rule_agent] = mixture_agent

        return self._produce_update(rule_embedding, mixture)

    def _produce_update(
        self, selection_map: dict[Agent, Agent], mixture: Mixture
    ) -> _MixtureUpdate:
        """Produce an update specification from selected agents.

        Args:
            selection_map: Mapping from rule agents to mixture agents.
            mixture: Current mixture state.
        """
        selection = [
            None if agent is None else selection_map[agent]
            for agent in self.left.agents
        ]  # Select agents in the mixture matching the rule, in order
        new_selection: list[Optional[Agent]] = [None] * len(
            selection
        )  # The new/modified agents used to make the appropriate edges
        update = _MixtureUpdate()

        # Manage agents
        for i in range(len(self)):
            l_agent = self.left.agents[i]
            r_agent = self.right.agents[i]
            agent: Optional[Agent] = selection[i]

            match l_agent, r_agent:
                case None, Agent():
                    new_selection[i] = update.create_agent(r_agent)
                case Agent(), None:
                    update.remove_agent(agent)
                case Agent(), Agent() if l_agent.type != r_agent.type:
                    update.remove_agent(agent)
                    new_selection[i] = update.create_agent(r_agent)
                case Agent(), Agent() if l_agent.type == r_agent.type:
                    for r_site in r_agent:
                        if r_site._stated:
                            update.set_site_state(agent[r_site.label], r_site.state)
                    new_selection[i] = agent
                case _:
                    pass

        # Manage explicitly referenced edges
        for i, r_agent in enumerate(self.right.agents):
            if r_agent is None:
                continue
            agent = new_selection[i]
            for r_site in r_agent:
                site = agent[r_site.label]
                match r_site.partner:
                    case Site() as r_partner:
                        partner_idx = self.right.agents.index(r_partner.agent)
                        partner = new_selection[partner_idx][r_partner.label]
                        update.connect_sites(site, partner)
                    case ".":
                        update.disconnect_site(site)
                    case x if (
                        x != "?"
                        and self.left.agents[i]
                        and x != self.left.agents[i][r_site.label].partner
                    ):
                        raise TypeError(
                            f"Site partners of type {x} are unsupported for right-hand rule patterns, unless they remain unchanged from the left-hand side."
                        )

        return update
