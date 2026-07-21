import random
from math import prod
from typing import Literal, Optional, Self, TYPE_CHECKING
from functools import cached_property
from copy import deepcopy

from pykappa.pattern import Pattern, Component, Agent, Site
from pykappa.mixture import Mixture, _MixtureUpdate
from pykappa._expression import Expression
from pykappa._utils import rejection_sample

if TYPE_CHECKING:
    from pykappa.system import System


# Useful constants
AVOGADRO = 6.02214e23
DIFFUSION_RATE = 1e9


class Rule:
    """A Kappa rule, specifying the transformation of a pattern at a stochastic rate."""

    left: Pattern
    right: Pattern
    rate_expression: Expression
    component_constraint: Literal["any", "same", "different"]
    token_updates: list[tuple[Expression, str]]

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

    def __init__(
        self,
        left: Pattern,
        right: Pattern,
        rate_expression: Expression,
        component_constraint: Literal["any", "same", "different"] = "any",
        token_updates: Optional[list[tuple[Expression, str]]] = None,
    ):
        self.left = left
        self.right = right
        self.rate_expression = rate_expression
        self.component_constraint = component_constraint
        self.token_updates = token_updates or []
        self._component_weights: dict[Component, int] = {}

        l = len(self.left.agents)
        r = len(self.right.agents)
        assert (
            l == r
        ), f"The left-hand side of this rule has {l} slots, but the right-hand side has {r}."
        assert component_constraint in {"any", "same", "different"}
        assert (
            component_constraint != "different" or len(self.left.components) == 2
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

    @property
    def kappa_str(self) -> str:
        token_part = ""
        if self.token_updates:
            updates_str = " ".join(
                f"{expr.kappa_str} {name}" for expr, name in self.token_updates
            )
            token_part = f" | {updates_str}"
        return f"{self.left.kappa_str} -> {self.right.kappa_str}{token_part} @ {self._rate_str}"

    def reactivity(self, system: "System") -> float:
        """Calculate the total reactivity of this rule in the given system,
        i.e. the number of embeddings times the reaction rate, accounting
        for rule symmetry.
        """
        return (
            self.n_embeddings(system.mixture) // self.n_symmetries * self.rate(system)
        )

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
                l_site = Site("__temp__", "?", partner=None)
                r_site = Site("__temp__", "?", partner=None)

                l_site.agent = l
                l_site.partner = r_site
                l_site.state = "left"
                l.interface["__temp__"] = l_site

                r_site.agent = r
                r_site.partner = l_site
                r_site.state = "right"
                r.interface["__temp__"] = r_site

        pattern = Pattern(left_agents + right_agents)
        return pattern.n_isomorphisms(pattern)

    def rate(self, system: "System") -> float:
        return self.rate_expression.evaluate(system)

    def n_embeddings(self, mixture: Mixture) -> int:
        """Count embeddings in the mixture.

        Note:
            This doesn't do any symmetry correction, though `System`
            applies this correction when calculating rule reactivities.
        """
        if self.component_constraint == "same":
            self._component_weights = {
                component: prod(
                    len(mixture.embeddings_in_component(pattern, component))
                    for pattern in self.left.components
                )
                for component in mixture.components
            }
            return sum(self._component_weights.values())

        if self.component_constraint == "different":
            first, second = self.left.components
            self._component_weights = {
                component: len(mixture.embeddings_in_component(first, component))
                * (
                    len(mixture.embeddings(second))
                    - len(mixture.embeddings_in_component(second, component))
                )
                for component in mixture.components
            }
            return sum(self._component_weights.values())

        return prod(
            len(mixture.embeddings(component)) for component in self.left.components
        )

    def _select(self, mixture: Mixture) -> Optional[_MixtureUpdate]:
        """Select agents and specify the update (or None for invalid match).

        Note:
            Can change the internal states of agents in the mixture but
            records everything else in the MixtureUpdate.
        """
        if self.component_constraint != "any":
            components = list(self._component_weights)
            selected_component = random.choices(
                components, [self._component_weights[c] for c in components]
            )[0]

            if self.component_constraint == "different":
                first, second = self.left.components
                return self._produce_update(
                    random.choice(
                        mixture.embeddings_in_component(first, selected_component)
                    )
                    | rejection_sample(
                        mixture.embeddings(second),
                        mixture.embeddings_in_component(second, selected_component),
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
            component_embedding = random.choice(component_embeddings)

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
                        if r_site.stated:
                            agent[r_site.label].state = r_site.state
                            if r_site.state != l_agent[r_site.label].state:
                                update.register_changed_agent(agent)
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
