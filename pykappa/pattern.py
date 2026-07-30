from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from itertools import permutations
from math import prod
from types import MappingProxyType
from typing import Self, Optional, Iterator, Iterable, Union, NamedTuple, TYPE_CHECKING

from pykappa.analysis import _ComponentPlot
from pykappa._utils import Counted, IndexedSet, IndexedSetView

if TYPE_CHECKING:
    from pykappa.mixture import Mixture


# String partner states can be: "#" (wildcard), "." (empty), "_" (bound), "?" (undetermined)
# "?" is the default in pattern instantiation and a wildcard in rules and observations
_TypedPartner = NamedTuple("_TypedPartner", [("site_name", str), ("agent_name", str)])
_Partner = str | _TypedPartner | int | Union["Site"]


class Site(Counted):
    """A site on an agent with state and binding partner information."""

    _agent: "Agent"
    _label: str
    _state: str
    _partner: _Partner

    def __init__(self, label: str, state: str, partner: _Partner):
        """
        Args:
            label: Name of the site.
            state: Internal state of the site.
            partner: Binding partner specification.
        """
        super().__init__()
        self._label = label
        self._state = state
        self._partner = partner

    def __repr__(self):
        return f'Site(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def agent(self) -> "Agent":
        """The agent this site belongs to."""
        return self._agent

    @property
    def label(self) -> str:
        """Name of the site."""
        return self._label

    @property
    def state(self) -> str:
        """Internal state of the site."""
        return self._state

    @property
    def partner(self) -> _Partner:
        """The binding partner."""
        return self._partner

    @property
    def _kappa_state_str(self) -> str:
        return "" if self.state == "?" else f"{{{self.state}}}"

    @property
    def kappa_str(self) -> str:
        partner_str = (
            ""
            if self.partner == "?"
            else "[_]" if self._coupled else f"[{self.partner}]"
        )
        return f"{self._label}{partner_str}{self._kappa_state_str}"

    @property
    def instantiable(self) -> bool:
        """Check if a concrete Site can be created from this pattern."""
        return not (
            self.state == "#"
            or self.partner in ("#", "_")
            or isinstance(self.partner, _TypedPartner)
        )

    @property
    def bound(self) -> bool:
        return self.partner == "_" or isinstance(self.partner, (_TypedPartner, Site))

    @property
    def _coupled(self) -> bool:
        """Check if the site is bound to a specific other site."""
        return isinstance(self.partner, Site)

    @property
    def _undetermined(self) -> bool:
        """Check if the site is in a state equivalent to leaving it unnamed in an agent."""
        return self.state == "?" and self.partner in ("?", ".")

    @property
    def _stated(self) -> bool:
        """Check if the site has a specific internal state."""
        return self.state not in ("#", "?")

    def _set_state(self, state: str) -> None:
        self._state = state

    def _set_partner(self, partner: _Partner) -> None:
        self._partner = partner

    def _embeds_in(self, other: Self) -> bool:
        """Check whether self as a pattern matches other as a concrete site."""
        if (self._stated and self.state != other.state) or (
            self.bound and not other._coupled
        ):
            return False

        match self.partner:
            case ".":
                return other.partner == "."
            case _TypedPartner():
                return (
                    self.partner.site_name == other.partner.label
                    and self.partner.agent_name == other.partner._agent.type
                )
            case Site():
                return (
                    self.partner._agent.type == other.partner._agent.type
                    and self.label == other.label
                )

        return True


class Agent(Counted):
    """Represents an agent with a type and collection of sites."""

    _type: str
    _interface: dict[str, Site]

    @staticmethod
    def neighborhood(agents: Iterable["Agent"], radius: int) -> set["Agent"]:
        """Get all agents within a distance radius of the given agents."""
        frontier = set(agents)
        seen = set(frontier)

        for _ in range(radius):
            frontier = {n for cur in frontier for n in cur.neighbors} - seen
            seen.update(frontier)
            if not frontier:
                break

        return seen

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a single agent from a Kappa string.

        Raises:
            AssertionError: If the string doesn't describe exactly one agent.
        """
        from pykappa._parsing import kappa_parser, KappaTransformer

        # Check pattern describes only a single agent
        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        assert len(input_tree.children) == 1
        pattern_tree = input_tree.children[0]
        assert pattern_tree.data == "pattern"
        assert (
            len(pattern_tree.children) == 1
        ), "Zero or more than one agent patterns were specified."
        agent_tree = pattern_tree.children[0]
        return KappaTransformer().transform(agent_tree)

    def __init__(self, type: str, sites: Iterable[Site]):
        super().__init__()
        self._type = type
        self._interface = {}
        for site in sites:
            self._add_site(site)

    def __iter__(self):
        yield from self._interface.values()

    def __getitem__(self, key: str) -> Site:
        return self._interface[key]

    def __repr__(self):
        return f'Agent(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def type(self) -> str:
        """Type name of the agent."""
        return self._type

    @property
    def interface(self) -> Mapping[str, Site]:
        """Maps site labels to sites."""
        return MappingProxyType(self._interface)

    def _add_site(self, site: Site) -> None:
        self._interface[site.label] = site
        site._agent = self

    @property
    def kappa_str(self):
        return f"{self._type}({" ".join(site.kappa_str for site in self)})"

    @property
    def instantiable(self) -> bool:
        """Check if a concrete Agent can be created from this pattern."""
        return all(site.instantiable for site in self)

    @property
    def neighbors(self) -> tuple[Self, ...]:
        """The agents directly connected to this one."""
        return tuple(site.partner.agent for site in self if site._coupled)

    @property
    def _depth_first_traversal(self) -> list[Self]:
        """Perform depth-first traversal starting from this agent."""
        visited = set()
        traversal = []
        stack = [self]
        while stack:
            if (agent := stack.pop()) not in visited:
                visited.add(agent)
                traversal.append(agent)
                stack.extend(agent.neighbors)
        return traversal

    def _detached(self) -> Self:
        """Create a clone with all sites emptied of partners."""
        return type(self)(
            self._type, [Site(site.label, site.state, ".") for site in self]
        )

    def _isomorphic(self, other: Self) -> bool:
        """Check if two Agents are equivalent locally, ignoring partners.

        Note:
            Doesn't assume agents of the same type will have the same site signatures.
        """
        if self._type != other._type:
            return False

        b_sites_leftover = set(other.interface)
        for site_name, a_site in self._interface.items():
            # Check that `b` has a site with the same name and state
            if site_name in other.interface:
                b_sites_leftover.remove(site_name)
                if a_site.state != other[site_name].state:
                    return False
            else:
                if not a_site._undetermined:
                    return False

        # Check that sites in `other` not mentioned in `self`are undetermined
        return all(other[site_name]._undetermined for site_name in b_sites_leftover)

    def _embeds_in(self, other: Self) -> bool:
        """Check whether self as a pattern matches other as a concrete agent."""
        if self._type != other._type:
            return False

        for a_site in self:
            if a_site.label not in other.interface and not a_site._undetermined:
                return False
            b_site = other[a_site.label]
            if not a_site._embeds_in(b_site):
                return False

        return True


class Embedding(dict[Agent, Agent]):
    """Dictionary representing a mapping from pattern agents to mixture agents."""

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __repr__(self):
        return f"Embedding({', '.join(f"{a.id}: {self[a].id}" for a in self)})"


class Component(Counted):
    """A set of agents that are all in the same connected component.

    Note:
        Connectedness is not guaranteed statically and must be enforced.
    """

    _agents: IndexedSet[Agent]

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a single component from a Kappa string.

        Raises:
            AssertionError: If the pattern doesn't represent exactly one component.
        """
        parsed_pattern = Pattern.from_kappa(kappa_str)
        assert len(parsed_pattern.components) == 1
        return parsed_pattern.components[0]

    def __init__(self, agents: Iterable[Agent]):
        """
        Raises:
            AssertionError: If agents list is empty.
        """
        super().__init__()
        assert agents
        self._agents = IndexedSet(agents)  # TODO: order by graph traversal
        self._agents.create_index("type", lambda a: [a.type])

        self.plot = _ComponentPlot(self)

    def __iter__(self):
        yield from self.agents

    @property
    def agents(self) -> IndexedSetView[Agent]:
        """The agents in this component."""
        return self._agents.view

    def __len__(self):
        return len(self.agents)

    def __repr__(self):
        return f'Component(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def kappa_str(self) -> str:
        return Pattern._agents_to_kappa_str(self.agents)

    def isomorphic(self, other: Self) -> bool:
        return next(self.isomorphisms(other), None) is not None

    def embeddings(
        self, other: Self | "Mixture" | Iterable[Agent], exact: bool = False
    ) -> Iterator[Embedding]:
        """Find embeddings of self in other.

        Args:
            other: Target to find embeddings in.
            exact: If True, finds isomorphisms instead of embeddings.
        """
        if hasattr(other, "agents"):
            other: IndexedSet[Agent] | IndexedSetView[Agent] = other.agents

        assert "type" in other.properties

        a_root = next(iter(self.agents))  # "a" refers to `self` and "b" to `other`
        # Narrow the search by mapping `a_root` to agents in `other` of the same type
        for b_root in other.lookup("type", a_root.type):

            agent_map = Embedding({a_root: b_root})  # The potential bijection
            frontier = {a_root}
            root_failed = False

            while frontier and not root_failed:
                a = frontier.pop()
                b = agent_map[a]

                match_func = a._isomorphic if exact else a._embeds_in
                if not match_func(b):
                    root_failed = True
                    break

                for a_site in a:
                    if a_site.label not in b.interface:
                        if not a_site._undetermined:
                            root_failed = True
                            break
                        else:
                            continue
                    b_site = b[a_site.label]

                    if a_site._coupled:
                        if not b_site._coupled:
                            root_failed = True
                            break

                        a_partner = a_site.partner.agent
                        b_partner = b_site.partner.agent

                        if b_partner not in other:
                            # The embedding must be enclosed in the given set of agents
                            root_failed = True
                            break
                        elif a_partner not in agent_map:
                            frontier.add(a_partner)
                            agent_map[a_partner] = b_partner
                        elif agent_map[a_site.partner.agent] != b_site.partner.agent:
                            root_failed = True
                            break
                    elif exact and a_site.partner != b_site.partner:
                        root_failed = True
                        break

            if not root_failed:
                yield agent_map  # A valid bijection

    def isomorphisms(self, other: Self | "Mixture") -> Iterator[dict[Agent, Agent]]:
        """Find bijections which respect links in the site graph.

        Checks for bijections ensuring that any internal site state specified
        in one component exists and is the same in the other.

        Note:
            Handles isomorphism generally, between instantiated components
            in a mixture and potentially between rule patterns.
        """
        if len(self.agents) != len(other.agents):
            return
        yield from self.embeddings(other, exact=True)

    @property
    def diameter(self) -> int:
        """Get the maximum minimum shortest path between any two agents."""

        def bfs_depth(root) -> int:
            frontier = set([root])
            seen = set()
            depth = -1

            while frontier:
                depth += 1
                new_frontier = set()
                seen = seen | frontier
                for cur in frontier:
                    for n in cur.neighbors:
                        if n not in seen:
                            new_frontier.add(n)

                frontier = new_frontier

            return depth

        return max(bfs_depth(a) for a in self.agents)


class Pattern:
    """A pattern consisting of multiple agents, some of which may be None (empty slots)."""

    _agents: list[Optional[Agent]]

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """
        Raises:
            AssertionError: If the string doesn't describe exactly one pattern.
        """
        from pykappa._parsing import kappa_parser, KappaTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        assert (
            len(input_tree.children) == 1
        ), "Zero or more than one patterns were specified."
        pattern_tree = input_tree.children[0]
        return KappaTransformer().transform(pattern_tree)

    def __init__(self, agents: Iterable[Optional[Agent]]):
        """
        Args:
            agents: Iterable of agents, where None represents empty slots.

        Raises:
            AssertionError: If integer links are malformed.
        """
        self._agents = tuple(agents)

        # Parse site connections implied by integer LinkStates
        integer_links: defaultdict[int, list[Site]] = defaultdict(list)
        for agent in self._agents:
            if agent is not None:
                for site in agent:
                    if isinstance(site.partner, int):
                        integer_links[site.partner].append(site)

        # Replace integer LinkStates with Agent references
        for i in integer_links:
            linked_sites = integer_links[i]
            if len(linked_sites) == 1:
                raise AssertionError(f"Site link {i} is only referenced in one site.")
            elif len(linked_sites) > 2:
                raise AssertionError(
                    f"Site link {i} is referenced in more than two sites."
                )
            else:
                linked_sites[0]._set_partner(linked_sites[1])
                linked_sites[1]._set_partner(linked_sites[0])

    def __iter__(self) -> Iterator[Optional[Agent]]:
        yield from self._agents

    def __len__(self):
        return len(self._agents)

    def __str__(self):
        return self.kappa_str

    @property
    def agents(self) -> tuple[Optional[Agent], ...]:
        """The agents in this pattern, including ``None`` empty slots."""
        return self._agents

    @cached_property
    def components(self) -> tuple[Component, ...]:
        """The connected components in this pattern."""
        unseen = {agent for agent in self._agents if agent is not None}
        components = []
        while unseen:
            component = Component(next(iter(unseen))._depth_first_traversal)
            unseen.difference_update(component)
            components.append(component)
        return tuple(components)

    @staticmethod
    def _agents_to_kappa_str(agents: Iterable[Optional[Agent]]) -> str:
        """Convert a collection of agents to Kappa string representation."""
        bond_num_counter = 1
        bond_nums: dict[Site, int] = dict()
        agent_strs = []
        for agent in agents:
            if agent is None:
                agent_strs.append(".")
                continue
            site_strs = []
            for site in agent:
                if site in bond_nums:
                    partner_str = f"[{bond_nums[site]}]"
                elif site._coupled:
                    partner_str = f"[{bond_num_counter}]"
                    bond_nums[site.partner] = bond_num_counter
                    bond_num_counter += 1
                else:
                    partner_str = "" if site.partner == "?" else f"[{site.partner}]"
                site_strs.append(f"{site.label}{partner_str}{site._kappa_state_str}")
            agent_strs.append(f"{agent.type}({" ".join(site_strs)})")
        return ", ".join(agent_strs)

    @property
    def kappa_str(self) -> str:
        return type(self)._agents_to_kappa_str(self._agents)

    @property
    def instantiable(self) -> bool:
        """Check if all agents in the pattern are specific enough to instantiate."""
        return all(agent is not None and agent.instantiable for agent in self._agents)

    def n_isomorphisms(self, other: Self) -> int:
        """Counts the number of bijections which respect links in the site graph.

        Note:
            Runtime is exponential in the number of components; use with caution.
        """
        if len(self.components) != len(other.components):
            return 0

        return sum(
            prod(
                len(list(left.isomorphisms(right)))
                for left, right in zip(self.components, perm)
            )
            for perm in permutations(other.components)
        )
