"""Represents Kappa rules."""

import random
from dataclasses import dataclass, field, replace
from math import prod
from typing import Literal, Mapping, Optional, Protocol, Self, TYPE_CHECKING
from functools import cached_property
from copy import deepcopy

from pykappa.pattern import Pattern, Component, Agent, Site
from pykappa.mixture import Mixture, _MixtureUpdate
from pykappa.expression import Expression
from pykappa._utils import rejection_sample

if TYPE_CHECKING:
    from pykappa.system import System


@dataclass(frozen=True)
class RuleMatch:
    """A concrete match of a rule's left-hand side in a mixture."""

    embedding: Mapping[Agent, Agent]
    components: tuple[Component, ...]


class RuleConstraint(Protocol):
    """A programmatic condition on a concrete rule match."""

    def accepts(self, match: RuleMatch, mixture: Mixture) -> bool: ...


class ComponentMatchConstraint:
    """A constraint that can filter each matched mixture component independently."""

    def accepts(self, match: RuleMatch, mixture: Mixture) -> bool:
        return all(self.accepts_component(component) for component in match.components)

    def accepts_component(self, component: Component) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ComponentSize(ComponentMatchConstraint):
    """Require every matched mixture component to contain at most ``max_size`` agents."""

    max_size: int

    def __post_init__(self):
        if self.max_size < 1:
            raise ValueError("max_size must be at least 1")

    def accepts_component(self, component: Component) -> bool:
        return len(component) <= self.max_size


@dataclass(frozen=True, eq=False)
class Rule:
    """A Kappa rule, specifying the transformation of a pattern at a stochastic rate."""

    left: Pattern
    right: Pattern
    rate_expression: Expression
    component_constraint: Literal["any", "same", "different"] = "any"
    token_updates: tuple[tuple[Expression, str], ...] = ()
    constraints: tuple[RuleConstraint, ...] = ()
    _component_weights: dict[Component, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _component_counts: dict[Component, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _coincident_weight: int = field(default=0, init=False, repr=False, compare=False)
    _component_totals: tuple[int, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    @classmethod
    def list_from_kappa(
        cls, kappa_str: str, constraints: tuple[RuleConstraint, ...] = ()
    ) -> list[Self]:
        """Parse Kappa string into a list of rules.

        Note:
            Forward-reverse rules (with "<->") represent two rules.
        """
        from pykappa._parsing import kappa_parser, KappaTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        rule_tree = input_tree.children[0]
        rules = KappaTransformer().transform(rule_tree)
        if constraints:
            return [replace(rule, constraints=constraints) for rule in rules]
        return rules

    @classmethod
    def from_kappa(
        cls, kappa_str: str, constraints: tuple[RuleConstraint, ...] = ()
    ) -> Self:
        """Parse a single Kappa rule from string.

        Raises:
            AssertionError: If the string represents more than one rule.
        """
        rules = cls.list_from_kappa(kappa_str, constraints)
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
        object.__setattr__(self, "constraints", tuple(self.constraints or ()))
        l = len(self.left.agents)
        r = len(self.right.agents)
        assert (
            l == r
        ), f"The left-hand side of this rule has {l} slots, but the right-hand side has {r}."
        assert self.component_constraint in {"any", "same", "different"}
        assert (
            self.component_constraint != "different" or len(self.left.components) == 2
        ), "A different-component constraint requires exactly 2 pattern components."

    @property
    def requires_component_tracking(self) -> bool:
        """Whether applying this rule requires connected-component tracking."""
        return self.component_constraint != "any" or bool(self.constraints)

    @cached_property
    def _component_match_constraints(self) -> tuple[ComponentMatchConstraint, ...]:
        return tuple(
            constraint
            for constraint in self.constraints
            if isinstance(constraint, ComponentMatchConstraint)
        )

    @cached_property
    def _uses_component_weights(self) -> bool:
        return self.component_constraint != "any" or bool(
            self._component_match_constraints
        )

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
            General constraints are applied during selection; component-match
            constraints are included in this count.
        """
        if self._uses_component_weights:
            self._component_counts.clear()
            self._component_weights.clear()
            totals = [0] * len(self.left.components)
            coincident_weight = 0
            for component in mixture.components:
                counts = self._eligible_counts_in_component(mixture, component)
                self._component_counts[component] = counts
                weight = prod(counts)
                self._component_weights[component] = weight
                coincident_weight += weight
                for i, count in enumerate(counts):
                    totals[i] += count
            object.__setattr__(self, "_component_totals", tuple(totals))
            object.__setattr__(self, "_coincident_weight", coincident_weight)
            return self._component_weight()

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

    def _eligible_counts_in_component(
        self, mixture: Mixture, component: Component
    ) -> tuple[int, ...]:
        if not all(
            constraint.accepts_component(component)
            for constraint in self._component_match_constraints
        ):
            return (0,) * len(self.left.components)
        return self._counts_in_component(mixture, component)

    def _component_weight(self) -> int:
        if self.component_constraint == "same":
            return self._coincident_weight
        if self.component_constraint == "different":
            return prod(self._component_totals) - self._coincident_weight
        return prod(self._component_totals)

    def update_component_weights(
        self,
        mixture: Mixture,
        previous_components: set[Component],
        current_components: set[Component],
    ) -> int:
        """Refresh constraint weights after an event touched some components."""
        totals = list(self._component_totals)
        coincident_weight = self._coincident_weight
        for component in previous_components:
            counts = self._component_counts.pop(
                component, (0,) * len(self.left.components)
            )
            weight = self._component_weights.pop(component, 0)
            coincident_weight -= weight
            for i, count in enumerate(counts):
                totals[i] -= count
        for component in current_components:
            counts = self._eligible_counts_in_component(mixture, component)
            weight = prod(counts)
            self._component_counts[component] = counts
            self._component_weights[component] = weight
            coincident_weight += weight
            for i, count in enumerate(counts):
                totals[i] += count
        object.__setattr__(self, "_component_totals", tuple(totals))
        object.__setattr__(self, "_coincident_weight", coincident_weight)
        return self._component_weight()

    def _select(
        self, mixture: Mixture, rng: random.Random | None = None
    ) -> Optional[_MixtureUpdate]:
        """Select agents and specify the update (or None for invalid match).

        Note:
            Can change the internal states of agents in the mixture but
            records everything else in the MixtureUpdate.
        """
        rng = random if rng is None else rng

        if self._uses_component_weights:
            components = list(mixture.components)
            if self.component_constraint == "different":
                second_total = self._component_totals[1]
                weights = [
                    first * (second_total - second)
                    for first, second in (
                        self._component_counts[component] for component in components
                    )
                ]
            elif self.component_constraint == "same":
                weights = [
                    self._component_weights[component] for component in components
                ]
            if self.component_constraint != "any":
                selected_component = rng.choices(components, weights)[0]

            if self.component_constraint == "different":
                first, second = self.left.components
                rule_embedding = dict(
                    rng.choice(
                        mixture.embeddings_in_component(first, selected_component)
                    )
                )
                if self._component_match_constraints:
                    second_component = rng.choices(
                        components,
                        [
                            0
                            if component == selected_component
                            else self._component_counts[component][1]
                            for component in components
                        ],
                    )[0]
                    second_embedding = rng.choice(
                        mixture.embeddings_in_component(second, second_component)
                    )
                else:
                    second_embedding = rejection_sample(
                        mixture.embeddings(second),
                        mixture.embeddings_in_component(second, selected_component),
                        rng=rng,
                    )
                rule_embedding.update(second_embedding)
                return self._constrained_update(rule_embedding, mixture)

            def embeddings(component, i):
                selected = (
                    selected_component
                    if self.component_constraint == "same"
                    else rng.choices(
                        components,
                        [self._component_counts[c][i] for c in components],
                    )[0]
                )
                return mixture.embeddings_in_component(component, selected)
        else:
            embeddings = lambda component, _: mixture.embeddings(component)

        rule_embedding: dict[Agent, Agent] = {}

        for i, component in enumerate(self.left.components):
            component_embeddings = (
                embeddings(component, i)
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

        return self._constrained_update(rule_embedding, mixture)

    def _constrained_update(
        self, rule_embedding: dict[Agent, Agent], mixture: Mixture
    ) -> Optional[_MixtureUpdate]:
        if not self.constraints:
            return self._produce_update(rule_embedding, mixture)
        components = tuple(
            mixture.components.lookup_one("agent", rule_embedding[next(iter(component))])
            for component in self.left.components
        )
        match = RuleMatch(rule_embedding, components)
        if not all(constraint.accepts(match, mixture) for constraint in self.constraints):
            return None
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
