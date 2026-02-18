from collections import defaultdict
from functools import cached_property
from itertools import permutations
from typing import Self, Optional, Iterator, Iterable, Union, NamedTuple, TYPE_CHECKING

from pykappa.utils import Counted, IndexedSet

if TYPE_CHECKING:
    from pykappa.mixture import Mixture


# String partner states can be: "#" (wildcard), "." (empty), "_" (bound), "?" (undetermined)
# "?" is the default in pattern instantiation and a wildcard in rules and observations
SiteType = NamedTuple("SiteType", [("site_name", str), ("agent_name", str)])
Partner = str | SiteType | int | Union["Site"]


class Site(Counted):
    """Represents a site on an agent with state and binding partner information.

    Attributes:
        agent: The agent this site belongs to (set after initialization).
        label: Name of the site.
        state: Internal state of the site.
        partner: Binding partner specification.
    """

    agent: "Agent"  # Expected to be set after initialization

    def __init__(self, label: str, state: str, partner: Partner):
        """Initialize a site with label, state, and partner.

        Args:
            label: Name of the site.
            state: Internal state of the site.
            partner: Binding partner specification.
        """
        super().__init__()
        self.label = label
        self.state = state
        self.partner = partner

    def __repr__(self):
        return f'Site(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def kappa_partner_str(self) -> str:
        if self.partner == "?":
            return ""
        elif self.coupled:
            return "[_]"
        return f"[{self.partner}]"

    @property
    def kappa_state_str(self) -> str:
        return "" if self.state == "?" else f"{{{self.state}}}"

    @property
    def kappa_str(self) -> str:
        """The site representation in Kappa format."""
        return f"{self.label}{self.kappa_partner_str}{self.kappa_state_str}"

    @property
    def undetermined(self) -> bool:
        """Check if the site is in a state equivalent to leaving it unnamed in an agent."""
        return self.state == "?" and self.partner in ("?", ".")

    @property
    def underspecified(self) -> bool:
        """Check if a concrete Site can be created from this pattern."""
        return (
            self.state == "#"
            or self.partner in ("#", "_")
            or isinstance(self.partner, SiteType)
        )

    @property
    def stated(self) -> bool:
        """Check if the site has a specific internal state."""
        return self.state not in ("#", "?")

    @property
    def bound(self) -> bool:
        """Check if the site is bound."""
        return (
            self.partner == "_"
            or isinstance(self.partner, SiteType)
            or isinstance(self.partner, Site)
        )

    @property
    def coupled(self) -> bool:
        """Check if the site is coupled to a specific other site."""
        return isinstance(self.partner, Site)

    def embeds_in(self, other: Self) -> bool:
        """Check whether self as a pattern matches other as a concrete site."""
        if (self.stated and self.state != other.state) or (
            self.bound and not other.coupled
        ):
            return False

        match self.partner:
            case ".":
                return other.partner == "."
            case SiteType():
                return (
                    self.partner.site_name == other.partner.label
                    and self.partner.agent_name == other.partner.agent.type
                )
            case Site():
                return (
                    self.partner.agent.type == other.partner.agent.type
                    and self.label == other.label
                )

        return True


class Agent(Counted):
    """Represents an agent with a type and collection of sites.

    Attributes:
        type: Type name of the agent.
        interface: Dictionary mapping site labels to Site objects.
    """

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a single agent from a Kappa string.

        Raises:
            AssertionError: If the string doesn't describe exactly one agent.
        """
        from pykappa.parsing import kappa_parser, KappaTransformer

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
        """Initialize an agent with given type and sites."""
        super().__init__()
        self.type = type
        self.interface = {site.label: site for site in sites}

    def __iter__(self):
        yield from self.sites

    def __getitem__(self, key: str) -> Site:
        """Get a site by its label."""
        return self.interface[key]

    def __repr__(self):
        return f'Agent(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def kappa_str(self):
        """The agent representation in Kappa format."""
        return f"{self.type}({" ".join(site.kappa_str for site in self)})"

    @property
    def sites(self) -> Iterable[Site]:
        """All sites of this agent."""
        yield from self.interface.values()

    @cached_property
    def underspecified(self) -> bool:
        """Check if a concrete Agent can be created from this pattern."""
        return any(site.underspecified for site in self)

    @property
    def neighbors(self) -> list[Self]:
        """The agents directly connected to this one."""
        return [site.partner.agent for site in self if site.coupled]

    @property
    def depth_first_traversal(self) -> list[Self]:
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

    @property
    def instantiable(self) -> bool:
        """Check if this agent pattern can be instantiated."""
        return not any(site.underspecified for site in self)

    def detached(self) -> Self:
        """Create a clone with all sites emptied of partners."""
        detached = type(self)(
            self.type, [Site(site.label, site.state, ".") for site in self]
        )
        for site in detached:
            site.agent = detached
        return detached

    def isomorphic(self, other: Self) -> bool:
        """Check if two Agents are equivalent locally, ignoring partners.

        Note:
            Doesn't assume agents of the same type will have the same site signatures.
        """
        if self.type != other.type:
            return False

        b_sites_leftover = set(other.interface)
        for site_name, a_site in self.interface.items():
            # Check that `b` has a site with the same name and state
            if site_name in other.interface:
                b_sites_leftover.remove(site_name)
                if a_site.state != other[site_name].state:
                    return False
            else:
                if not a_site.undetermined:
                    return False

        # Check that sites in `other` not mentioned in `self`are undetermined
        return all(other[site_name].undetermined for site_name in b_sites_leftover)

    def embeds_in(self, other: Self) -> bool:
        """Check whether self as a pattern matches other as a concrete agent."""
        if self.type != other.type:
            return False

        for a_site in self:
            if a_site.label not in other.interface and not a_site.undetermined:
                return False
            b_site = other[a_site.label]
            if not a_site.embeds_in(b_site):
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

    agents: IndexedSet[Agent]
    n_copies: int

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a single component from a Kappa string.

        Raises:
            AssertionError: If the pattern doesn't represent exactly one component.
        """
        parsed_pattern = Pattern.from_kappa(kappa_str)
        assert len(parsed_pattern.components) == 1
        return parsed_pattern.components[0]

    def __init__(self, agents: Iterable[Agent], n_copies: int = 1):
        """Initialize a component with agents.

        Raises:
            AssertionError: If agents list is empty or n_copies < 1.
            NotImplementedError: If n_copies != 1 (not yet supported).
        """
        super().__init__()
        assert agents
        assert n_copies >= 1
        if n_copies != 1:
            raise NotImplementedError(
                "Simulations won't handle n_copies correctly in counting embeddings."
            )

        self.agents = IndexedSet(agents)  # TODO: order by graph traversal
        self.agents.create_index("type", lambda a: [a.type])
        self.n_copies = n_copies

    def __iter__(self):
        yield from self.agents

    def __len__(self):
        return len(self.agents)

    def __repr__(self):
        return f'Component(id={self.id}, kappa_str="{self.kappa_str}")'

    @property
    def kappa_str(self) -> str:
        """The component representation in Kappa format."""
        return Pattern.agents_to_kappa_str(self.agents)

    def add(self, agent: Agent):
        """Add an agent to this component."""
        self.agents.add(agent)

    def isomorphic(self, other: Self) -> bool:
        """Check if two components are isomorphic."""
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
            other: IndexedSet[Agent] = other.agents

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

                match_func = a.isomorphic if exact else a.embeds_in
                if not match_func(b):
                    root_failed = True
                    break

                for a_site in a:
                    if a_site.label not in b.interface:
                        if not a_site.undetermined:
                            root_failed = True
                            break
                        else:
                            continue
                    b_site = b[a_site.label]

                    if a_site.coupled:
                        if not b_site.coupled:
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

    @cached_property
    def n_automorphisms(self) -> int:
        """Returns the number of automorphisms (i.e. isomorphisms onto itself)
        of the component.

        Note:
            This uses a cached result, and thus should only be used
            for static components, i.e. the ones in rules and observables.
            Do not use this for components in a mixture that can change.
        """
        return len(list(self.isomorphisms(self)))

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

    def plot(self, legend: bool = True) -> None:
        """Plot the component using graphviz."""
        import colorsys
        from graphviz import Source

        agent_types = list(dict.fromkeys(a.type for a in self.agents))
        type_color = {
            t: "#{:02x}{:02x}{:02x}".format(
                *[
                    int(c * 255)
                    for c in colorsys.hls_to_rgb(i / max(len(agent_types), 1), 0.4, 0.8)
                ]
            )
            for i, t in enumerate(agent_types)
        }

        edges = set()
        for a in self.agents:
            for b in a.neighbors:
                if a is b:
                    continue
                edges.add(tuple(sorted((id(a), id(b)))))

        lines = [
            "graph {",
            "  graph [overlap=false, splines=false, nodesep=0.05, ranksep=0.05];",
            "  layout=twopi;",  # radial layout
            '  node  [shape=circle, width=0.05, height=0.05, fixedsize=true, label="", style=filled];',
            "  edge  [penwidth=0.3];",
        ]
        if legend:
            lines += [
                "  subgraph cluster_legend {",
                '    label="Legend"; style=filled; fillcolor=white; fontsize=10;',
                "    node [shape=circle, width=0.15, height=0.15, fixedsize=true, style=filled, fontsize=8, labelloc=r];",
            ]
            for t, color in type_color.items():
                lines.append(f'    legend_{t} [label="{t}", fillcolor="{color}"];')
            lines.append("  }")
        for a in self.agents:
            color = type_color[a.type]
            lines.append(f'  a{id(a)} [fillcolor="{color}"];')
        for u, v in edges:
            lines.append(f"  a{u} -- a{v};")
        lines.append("}")

        return Source("\n".join(lines), engine="twopi")


class Pattern:
    """A pattern consisting of multiple agents, some of which may be None (empty slots).

    Attributes:
        agents: List of agents, where None represents empty slots in rules.
    """

    agents: list[Optional[Agent]]

    @classmethod
    def from_kappa(cls, kappa_str: str) -> Self:
        """Parse a pattern from a Kappa string.

        Args:
            kappa_str: Kappa string describing a pattern.

        Returns:
            Parsed Pattern object.

        Raises:
            AssertionError: If the string doesn't describe exactly one pattern.
        """
        from pykappa.parsing import kappa_parser, KappaTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        assert (
            len(input_tree.children) == 1
        ), "Zero or more than one patterns were specified."
        assert len(input_tree.children) == 1
        pattern_tree = input_tree.children[0]
        return KappaTransformer().transform(pattern_tree)

    def __init__(self, agents: list[Optional[Agent]]):
        """Compile a pattern from a list of Agents.

        Replaces integer link states with references to actual partners, and
        constructs helper objects for tracking connected components. A None
        in agents represents an empty slot in a rule expression pattern.

        Args:
            agents: List of agents, where None represents empty slots.

        Raises:
            AssertionError: If integer links are malformed.
        """
        self.agents = agents

        # Parse site connections implied by integer LinkStates
        integer_links: defaultdict[int, list[Site]] = defaultdict(list)
        for agent in agents:
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
                linked_sites[0].partner = linked_sites[1]
                linked_sites[1].partner = linked_sites[0]

    def __iter__(self) -> Iterator[Optional[Agent]]:
        yield from self.agents

    def __len__(self):
        return len(self.agents)

    @cached_property
    def components(self) -> list[Component]:
        """The connected components in this pattern."""
        unseen = set(agent for agent in self.agents if agent is not None)
        components = []
        while unseen:
            component = Component(next(iter(unseen)).depth_first_traversal)
            unseen = unseen.difference(component)
            components.append(component)
        return components

    @staticmethod
    def agents_to_kappa_str(agents: Iterable[Optional[Agent]]) -> str:
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
                elif site.coupled:
                    partner_str = f"[{bond_num_counter}]"
                    bond_nums[site.partner] = bond_num_counter
                    bond_num_counter += 1
                else:
                    partner_str = "" if site.partner == "?" else f"[{site.partner}]"
                site_strs.append(f"{site.label}{partner_str}{site.kappa_state_str}")
            agent_strs.append(f"{agent.type}({" ".join(site_strs)})")
        return ", ".join(agent_strs)

    @property
    def kappa_str(self) -> str:
        """The pattern representation in Kappa format."""
        return type(self).agents_to_kappa_str(self.agents)

    @cached_property
    def underspecified(self) -> bool:
        """Check if any agents in the pattern are underspecified."""
        return any(agent is None or agent.underspecified for agent in self.agents)

    def n_isomorphisms(self, other: Self) -> int:
        """Counts the number of bijections which respect links in the site graph.

        Note:
            Runtime is exponential in the number of components; use with caution.
        """
        if len(self.components) != len(other.components):
            return 0

        res = 0
        for perm in permutations(other.components):
            temp = 1
            for l, r in zip(self.components, perm):
                temp *= len(list(l.isomorphisms(r)))
            res += temp
        return res
