import random
from math import prod
from abc import ABC, abstractmethod
from typing import Optional, Self, TYPE_CHECKING
from functools import cached_property
from copy import deepcopy

from pykappa.pattern import Pattern, Component, Agent, Site
from pykappa.mixture import Mixture, MixtureUpdate
from pykappa.expression import Expression
from pykappa.utils import rejection_sample

if TYPE_CHECKING:
    from pykappa.system import System


# Useful constants
AVOGADRO = 6.02214e23
DIFFUSION_RATE = 1e9
KDS = {"weak": 1e-6, "moderate": 1e-7, "strong": 1e-8}
VOLUMES = {"fibro": 2.25e-12, "yeast": 4.2e-14}
ROOM_TEMPERATURE = 273.15 + 25


def kinetic_to_stochastic_on_rate(
    k_on: float = DIFFUSION_RATE, volume: float = 1, molecularity: int = 2
) -> float:
    """Convert a kinetic on-rate constant to a stochastic one.

    Args:
        k_on: Kinetic on-rate constant.
        volume: Reaction volume.
        molecularity: Number of reactants.
    """
    return k_on / (AVOGADRO * volume ** (molecularity - 1))


class Rule(ABC):
    """Abstract base class for all rule types."""

    def reactivity(self, system: "System") -> float:
        """Calculate the total reactivity of this rule in the given system."""
        return self.n_embeddings(system.mixture) * self.rate(system)

    @abstractmethod
    def rate(self, system: "System") -> float:
        """Get the stochastic rate of the rule."""
        pass

    @abstractmethod
    def n_embeddings(self, mixture: Mixture) -> int:
        """Count the number of ways this rule can be applied to the mixture."""
        pass

    @abstractmethod
    def select(self, mixture: Mixture) -> Optional[MixtureUpdate]:
        """Select agents in the mixture and specify the update (or None for null
        event).

        Note:
            Don't modify anything in mixture directly except for changing
            internal sites of agents. A null event is represented by returning None.
        """
        pass


class KappaRule(Rule):
    """Standard Kappa rule with left-hand side, right-hand side, and rate.

    Attributes:
        left: Left-hand side pattern.
        right: Right-hand side pattern.
        stochastic_rate: Rate expression for the rule.
    """

    left: Pattern
    right: Pattern
    stochastic_rate: Expression

    @classmethod
    def list_from_kappa(cls, kappa_str: str) -> list[Self]:
        """Parse Kappa string into a list of rules.

        Note:
            Forward-reverse rules (with "<->") represent two rules.
        """
        from pykappa.parsing import kappa_parser, KappaTransformer

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

    def __init__(self, left: Pattern, right: Pattern, stochastic_rate: Expression):
        self.left = left
        self.right = right
        self.stochastic_rate = stochastic_rate

    def __post_init__(self):
        l = len(self.left.agents)
        r = len(self.right.agents)
        assert (
            l == r
        ), f"The left-hand side of this rule has {l} slots, but the right-hand side has {r}."

    def __len__(self):
        return len(self.left.agents)

    def __iter__(self):
        yield from zip(self.left.agents, self.right.agents)

    def __repr__(self):
        return f'{type(self).__name__}(kappa_str="{self.kappa_str}")'

    def __str__(self):
        return self.kappa_str

    @property
    def kappa_str(self) -> str:
        """The rule representation in Kappa format."""
        return f"{self.left.kappa_str} -> {self.right.kappa_str} @ {self.stochastic_rate.kappa_str}"

    def reactivity(self, system: "System") -> float:
        """Calculate the total reactivity of this rule in the given system,
        i.e. the number of embeddings times the reaction rate, accounting
        for rule symmetry.
        """
        n_embeddings = self.n_embeddings(system.mixture)
        n_symmetries = self.n_symmetries
        assert n_embeddings % n_symmetries == 0
        return n_embeddings // n_symmetries * self.rate(system)

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
        """Evaluate the stochastic rate expression."""
        return self.stochastic_rate.evaluate(system)

    def n_embeddings(self, mixture: Mixture) -> int:
        """Count embeddings in the mixture.

        Note:
            This doesn't do any symmetry correction, though `System`
            applies this correction when calculating rule reactivities.
        """
        return prod(
            len(mixture.embeddings(component)) for component in self.left.components
        )

    def select(self, mixture: Mixture) -> Optional[MixtureUpdate]:
        """Select agents in the mixture and specify the update (or None
        for invalid match).

        Note:
            Can change the internal states of agents in the mixture but
            records everything else in the MixtureUpdate.
        """
        rule_embedding: dict[Agent, Agent] = {}

        for component in self.left.components:
            component_embeddings = mixture.embeddings(component)
            assert (
                len(component_embeddings) > 0
            ), f"A rule with no valid embeddings was selected: {self}"
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
    ) -> MixtureUpdate:
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
        update = MixtureUpdate()

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


class KappaRuleUnimolecular(KappaRule):
    """Unimolecular Kappa rule that acts within a single component.

    Attributes:
        component_weights: Cache of embedding weights per component.
    """

    def __post_init__(self):
        """Initialize the rule and component weights cache."""
        super().__post_init__()
        self.component_weights: dict[Component, int] = {}

    @property
    def kappa_str(self) -> str:
        """Get the rule representation in Kappa format."""
        return f"{self.left.kappa_str} -> {self.right.kappa_str} @ 0 {{{self.stochastic_rate.kappa_str}}}"

    def n_embeddings(self, mixture: Mixture) -> int:
        """Count the total number of embeddings in the mixture."""
        count = 0
        self.component_weights = {}
        for component in mixture.components:
            weight = prod(
                len(mixture.embeddings_in_component(match_component, component))
                for match_component in self.left.components
            )
            self.component_weights[component] = weight
            count += weight
        return count

    def select(self, mixture: Mixture) -> Optional[MixtureUpdate]:
        """Select agents in the mixture and specify the update (or None
        for invalid match).

        Note:
            n_embeddings must be called before this method so that the
            component_weights cache is up-to-date.
        """
        components_ordered = list(self.component_weights)
        weights = [self.component_weights[c] for c in components_ordered]
        selected_component = random.choices(components_ordered, weights)[0]

        selection_map: dict[Agent, Agent] = {}
        for component in self.left.components:
            choices = mixture.embeddings_in_component(component, selected_component)
            assert (
                len(choices) > 0
            ), f"A rule with no valid embeddings was selected: {self}"
            component_selection = random.choice(choices)

            for agent in component_selection:
                if component_selection[agent] in selection_map.values():
                    return None
                else:
                    selection_map[agent] = component_selection[agent]

        return self._produce_update(selection_map, mixture)


class KappaRuleBimolecular(KappaRule):
    """Bimolecular Kappa rule.

    Attributes:
        component_weights: Cache of embedding weights per component.
    """

    def __post_init__(self):
        """Initialize the rule and validate it has exactly 2 components."""
        super().__post_init__()
        self.component_weights: dict[Component, int] = {}
        assert (
            len(self.left.components) == 2
        ), "Bimolecular rule patterns must consist of exactly 2 components."

    @property
    def kappa_str(self) -> str:
        """The rule representation in Kappa format."""
        return super().kappa_str + "{0}"

    def n_embeddings(self, mixture: Mixture) -> int:
        """Count the total number of embeddings in the mixture."""
        count = 0
        self.component_weights = {}

        for component in mixture.components:
            n_match1 = len(
                mixture.embeddings_in_component(self.left.components[0], component)
            )
            n_match2 = len(mixture.embeddings(self.left.components[1])) - len(
                mixture.embeddings_in_component(self.left.components[1], component)
            )

            weight = n_match1 * n_match2
            self.component_weights[component] = weight
            count += weight

        return count

    def select(self, mixture: Mixture) -> Optional[MixtureUpdate]:
        """Select agents in the mixture and specify the update (or None
        for invalid match).

        Note:
            n_embeddings must be called before this method so that the
            component_weights cache is up-to-date.
        """
        components_ordered = list(self.component_weights.keys())
        weights = [self.component_weights[c] for c in components_ordered]
        selected_component = random.choices(components_ordered, weights)[0]

        match1 = random.choice(
            mixture.embeddings_in_component(self.left.components[0], selected_component)
        )
        match2 = rejection_sample(
            mixture.embeddings(self.left.components[1]),
            mixture.embeddings_in_component(
                self.left.components[1], selected_component
            ),
        )

        return self._produce_update(match1 | match2, mixture)
