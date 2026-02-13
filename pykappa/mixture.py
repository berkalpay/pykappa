from dataclasses import dataclass, field
from typing import Optional, Iterable, Iterator, Self

from pykappa.pattern import Site, Agent, Component, Pattern, Embedding
from pykappa.utils import IndexedSet


@dataclass(frozen=True)
class Edge:
    """Represents bonds between sites.

    Note:
        Edge(x, y) is the same as Edge(y, x).

    Attributes:
        site1: First site in the bond.
        site2: Second site in the bond.
    """

    site1: Site
    site2: Site

    def __eq__(self, other):
        return (self.site1 == other.site1 and self.site2 == other.site2) or (
            self.site1 == other.site2 and self.site2 == other.site1
        )

    def __hash__(self):
        return hash(frozenset((self.site1, self.site2)))


@dataclass
class Mixture:
    """A collection of agents and their connections.

    Attributes:
        agents: Indexed set of all agents in the mixture.
        _embeddings: Cache of embeddings for tracked components.
        _max_embedding_width: Maximum diameter of tracked components.
    """

    agents: IndexedSet[Agent]
    _embeddings: dict[Component, IndexedSet[Embedding]]
    _max_embedding_width: int

    @classmethod
    def from_kappa(cls, patterns: dict[str, int]) -> Self:
        """Create a mixture from Kappa pattern strings and counts.

        Args:
            patterns: Dictionary mapping pattern strings to copy counts.

        Returns:
            New Mixture with instantiated patterns.
        """
        real_patterns = []
        for pattern, count in patterns.items():
            real_patterns.extend([Pattern.from_kappa(pattern)] * count)
        return cls(real_patterns)

    def __init__(self, patterns: Optional[Iterable[Pattern]] = None):
        """Initialize a new mixture.

        Args:
            patterns: Optional collection of patterns to instantiate.
        """
        self.agents = IndexedSet()
        self._embeddings = {}
        self._max_embedding_width = 0

        self.agents.create_index("type", lambda a: [a.type])

        if patterns is not None:
            for pattern in patterns:
                self.instantiate(pattern)

    def __iter__(self) -> Iterator[Component]:
        yield from ComponentMixture([Pattern(list(self.agents))])

    @property
    def kappa_str(self) -> str:
        """The mixture representation in Kappa format.

        Returns:
            Kappa string with %init declarations for each component type.
        """
        return "\n".join(
            f"%init: {len(components)} {group.kappa_str}"
            for group, components in grouped(
                list(component for component in self)
            ).items()
        )

    def instantiate(self, pattern: Pattern | str, n_copies: int = 1) -> None:
        """Add instances of a pattern to the mixture.

        Args:
            pattern: Pattern to instantiate, or Kappa string.
            n_copies: Number of copies to create.

        Raises:
            AssertionError: If pattern is underspecified.
        """
        if isinstance(pattern, str):
            pattern = Pattern.from_kappa(pattern)

        assert (
            not pattern.underspecified
        ), "Pattern isn't specific enough to instantiate."
        for _ in range(n_copies):
            for component in pattern.components:
                self.add(component)

    def add(self, component: Component) -> None:
        """Add a component to the mixture.

        Args:
            component: Component to add with its agents and connections.
        """
        component_ordered = list(component.agents)
        new_agents = [agent.detached() for agent in component_ordered]
        new_edges = set()

        for i, agent in enumerate(component_ordered):
            # Duplicate the proper link structure
            for site in agent:
                if site.coupled:
                    partner = site.partner
                    i_partner = component_ordered.index(partner.agent)
                    new_site = new_agents[i][site.label]
                    new_partner = new_agents[i_partner][partner.label]
                    new_edges.add(Edge(new_site, new_partner))

        update = MixtureUpdate(agents_to_add=new_agents, edges_to_add=new_edges)
        self.apply_update(update)

    def remove(self, component: Component) -> None:
        """Remove a component from the mixture.

        Args:
            component: Component to remove.
        """
        update = MixtureUpdate()
        for agent in component:
            update.remove_agent(agent)
        self.apply_update(update)

    def embeddings(self, component: Component) -> IndexedSet[Embedding]:
        """Get embeddings of a tracked component.

        Notes:
            Returns the number of matches directly returned
            by subgraph isomorphism, i.e. not accounting for symmetries.

        Args:
            component: Component to get embeddings for.

        Returns:
            Set of embeddings for the component.

        Raises:
            KeyError: If component is not being tracked.
        """
        try:
            return self._embeddings[component]
        except KeyError as e:
            e.add_note(
                f"Undeclared component: {component}. To embed it, first use `track_component`."
            )
            raise

    def track_component(self, component: Component):
        """Start tracking embeddings of a component.

        Args:
            component: Component pattern to track.
        """
        self._max_embedding_width = max(component.diameter, self._max_embedding_width)
        embeddings = IndexedSet(component.embeddings(self))
        embeddings.create_index("agent", lambda e: iter(e.values()))
        self._embeddings[component] = embeddings

    def apply_update(self, update: "MixtureUpdate") -> None:
        """Apply a collection of changes to the mixture.

        Args:
            update: MixtureUpdate specifying changes to apply.
        """
        for agent in update.touched_before:
            for tracked in self._embeddings:
                self._embeddings[tracked].remove_by("agent", agent)

        for edge in update.edges_to_remove:
            self._remove_edge(edge)
        for agent in update.agents_to_remove:
            self._remove_agent(agent)
        for agent in update.agents_to_add:
            self._add_agent(agent)
        for edge in update.edges_to_add:
            self._add_edge(edge)
        # NOTE: the current implementation doesn't directly mutate agent type

        update_region = neighborhood(update.touched_after, self._max_embedding_width)

        update_region = IndexedSet(update_region)
        update_region.create_index("type", lambda a: [a.type])
        for component_pattern in self._embeddings:
            new_embeddings = component_pattern.embeddings(update_region)
            for e in new_embeddings:
                self._embeddings[component_pattern].add(e)

    def _add_agent(self, agent: Agent) -> None:
        """Add an agent to the mixture.

        Note:
            Calling these private functions isn't guaranteed to keep indexes
            up to date, which is why they shouldn't be used externally.
            The provided agent should not have any bound sites.

        Args:
            agent: Agent to add (should have empty sites).

        Raises:
            AssertionError: If agent has bound sites or isn't instantiable.
        """
        assert all(site.partner == "." for site in agent)  # Check all sites are unbound
        assert agent.instantiable
        self.agents.add(agent)

    def _remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the mixture.

        Note:
            Any bonds associated with agent must be removed first.

        Args:
            agent: Agent to remove (should have empty sites).

        Raises:
            AssertionError: If agent has bound sites.
        """
        assert all(site.partner == "." for site in agent)  # Check all sites are unbound
        self.agents.remove(agent)

    def _add_edge(self, edge: Edge) -> None:
        """Add a bond between two sites.

        Args:
            edge: Edge specifying the bond to create.

        Raises:
            AssertionError: If either agent is not in the mixture.
        """
        assert edge.site1.agent in self.agents
        assert edge.site2.agent in self.agents
        edge.site1.partner = edge.site2
        edge.site2.partner = edge.site1

    def _remove_edge(self, edge: Edge) -> None:
        """Remove a bond between two sites.

        Args:
            edge: Edge specifying the bond to remove.

        Raises:
            AssertionError: If the edge doesn't exist.
        """
        assert edge.site1.partner == edge.site2
        assert edge.site2.partner == edge.site1
        edge.site1.partner = "."
        edge.site2.partner = "."


@dataclass
class ComponentMixture(Mixture):
    """A mixture that explicitly tracks connected components.

    Attributes:
        components: Indexed set of all components in the mixture.
    """

    components: IndexedSet[Component]

    def __init__(self, patterns: Optional[Iterable[Pattern]] = None):
        """Initialize a component-tracking mixture.

        Args:
            patterns: Optional collection of patterns to instantiate.
        """
        self.components = IndexedSet()
        self.components.create_index("agent", lambda c: c.agents)
        super().__init__(patterns)

    def __iter__(self) -> Iterator[Component]:
        yield from self.components

    def embeddings_in_component(
        self, match_pattern: Component, mixture_component: Component
    ) -> list[dict[Agent, Agent]]:
        """Get embeddings of a pattern within a specific component.

        Args:
            match_pattern: Pattern to find embeddings for.
            mixture_component: Component to search within.

        Returns:
            List of embeddings within the specified component.
        """
        return self._embeddings[match_pattern].lookup("component", mixture_component)

    def track_component(self, component: Component):
        """Start tracking embeddings of a component pattern.

        Args:
            component: Component pattern to track.
        """
        super().track_component(component)
        self._embeddings[component].create_index(
            "component",
            lambda e: [self.components.lookup_one("agent", next(iter(e.values())))],
        )

    def _add_agent(self, agent: Agent) -> None:
        """Add an agent as a new single-agent component.

        Args:
            agent: Agent to add.
        """
        super()._add_agent(agent)
        component = Component([agent])
        self.components.add(component)

    def _remove_agent(self, agent: Agent) -> None:
        """Remove an agent and its component.

        Args:
            agent: Agent to remove.

        Raises:
            AssertionError: If agent is part of a multi-agent component.
        """
        super()._remove_agent(agent)
        component = self.components.lookup_one("agent", agent)
        self.components.remove(component)

    def _add_edge(self, edge: Edge) -> None:
        """Add an edge, potentially merging components.

        Args:
            edge: Edge to add between sites.
        """
        super()._add_edge(edge)

        # If the agents are in different components, merge the components
        # TODO: incremental mincut
        component1 = self.components.lookup_one("agent", edge.site1.agent)
        component2 = self.components.lookup_one("agent", edge.site2.agent)
        if component1 == component2:
            return

        # Ensure `component2` is the smaller of the 2
        if len(component2) > len(component1):
            component1, component2 = component2, component1

        relocated: dict[Component, list[Embedding]] = {}
        for tracked in self._embeddings:
            relocated[tracked] = list(
                self._embeddings[tracked].lookup("component", component2)
            )
            for e in relocated[tracked]:
                self._embeddings[tracked].remove(e)

        self.components.remove(component2)  # NOTE: invokes a redundant linear time pass
        for agent in component2:
            component1.add(agent)
            # TODO: better semantics for this type of operation
            #       Operate on diffs to set property.. ?
            self.components.indices["agent"][agent] = [component1]

        for tracked in self._embeddings:
            # TODO: refactor when we can register IndexedSet item updates, including
            # cached property evaluations
            for e in relocated[tracked]:
                assert (
                    self.components.lookup_one("agent", next(iter(e.values())))
                    == component1
                )
                self._embeddings[tracked].add(e)

    def _remove_edge(self, edge: Edge) -> None:
        """Remove an edge, potentially splitting components.

        Args:
            edge: Edge to remove.
        """
        super()._remove_edge(edge)

        agent1: Agent = edge.site1.agent
        agent2: Agent = edge.site2.agent
        old_component = self.components.lookup_one("agent", agent1)
        assert old_component == self.components.lookup_one("agent", agent2)

        # Create a new component if the old one got disconnected
        maybe_new_component = Component(agent1.depth_first_traversal)

        if agent2 in maybe_new_component:
            return  # The old component is still connected, do nothing

        new_component1 = maybe_new_component
        new_component2 = Component(agent2.depth_first_traversal)

        relocated: dict[Component, list[Embedding]] = {}
        for tracked in self._embeddings:
            relocated[tracked] = list(
                self._embeddings[tracked].lookup("component", old_component)
            )
            for e in relocated[tracked]:
                self._embeddings[tracked].remove(e)

        # TODO: need to do manual updates to the indices in `components`
        # to do this more efficiently
        self.components.remove(old_component)
        self.components.add(new_component1)
        self.components.add(new_component2)

        for tracked in self._embeddings:
            # TODO: refactor when we can register IndexedSet item updates, including
            # cached property evaluations
            for e in relocated[tracked]:
                assert self.components.lookup_one("agent", next(iter(e.values()))) in [
                    new_component1,
                    new_component2,
                ]
                self._embeddings[tracked].add(e)


@dataclass
class MixtureUpdate:
    """Specifies changes to be applied to a mixture.

    Attributes:
        agents_to_add: Agents to be added to the mixture.
        agents_to_remove: Agents to be removed from the mixture.
        edges_to_add: Edges to be created.
        edges_to_remove: Edges to be removed.
        agents_changed: Agents with internal state changes.
    """

    agents_to_add: list[Agent] = field(default_factory=list)
    agents_to_remove: list[Agent] = field(default_factory=list)
    edges_to_add: set[Edge] = field(default_factory=set)
    edges_to_remove: set[Edge] = field(default_factory=set)
    agents_changed: set[Agent] = field(default_factory=set)  # Agents changed internally

    def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent based on a template.

        Note:
            Sites in the created agent will be emptied.

        Args:
            agent: Template agent to base the new agent on.

        Returns:
            New agent with empty sites.
        """
        new_agent = agent.detached()
        self.agents_to_add.append(new_agent)
        return new_agent

    def remove_agent(self, agent: Agent) -> None:
        """Specify to remove an agent and its edges from the mixture.

        Args:
            agent: Agent to remove.
        """
        self.agents_to_remove.append(agent)
        for site in agent:
            if site.coupled:
                self.edges_to_remove.add(Edge(site, site.partner))

    def connect_sites(self, site1: Site, site2: Site) -> None:
        """Specify to create an edge between two sites.

        If the sites are bound to other sites, indicates to remove those edges.

        Args:
            site1: First site to connect.
            site2: Second site to connect.
        """
        if site1.coupled and site1.partner != site2:
            self.disconnect_site(site1)
        if site2.coupled and site2.partner != site1:
            self.disconnect_site(site2)
        if not site1.partner == site2:
            self.edges_to_add.add(Edge(site1, site2))

    def disconnect_site(self, site: Site) -> None:
        """Specify that a site should be unbound.

        Args:
            site: Site to disconnect from its partner.
        """
        if site.coupled:
            self.edges_to_remove.add(Edge(site, site.partner))

    def register_changed_agent(self, agent: Agent) -> None:
        """Register an agent as having internal state changes.

        Args:
            agent: Agent that has been internally modified.
        """
        self.agents_changed.add(agent)

    @property
    def touched_after(self) -> set[Agent]:
        """The agents that will be changed or added after this update.

        Returns:
            Set of agents affected by the update.
        """
        touched = self.agents_changed | set(self.agents_to_add)

        for edge in self.edges_to_add:
            touched.add(edge.site1.agent)
            touched.add(edge.site2.agent)

        for edge in self.edges_to_remove:
            a, b = edge.site1.agent, edge.site2.agent
            if a not in self.agents_to_remove:  # TODO make agents_to_remove a set
                touched.add(a)
            if b not in self.agents_to_remove:
                touched.add(b)

        return touched

    @property
    def touched_before(self) -> set[Agent]:
        """The agents that will be changed or removed by this update.

        Returns:
            Set of agents affected before the update is applied.
        """
        touched = self.agents_changed | set(self.agents_to_remove)

        for edge in self.edges_to_remove:
            touched.add(edge.site1.agent)
            touched.add(edge.site2.agent)

        for edge in self.edges_to_add:
            a, b = edge.site1.agent, edge.site2.agent
            if a not in self.agents_to_add:  # TODO make agents_to_add a set
                touched.add(a)
            if b not in self.agents_to_add:
                touched.add(b)

        return touched


def neighborhood(agents: Iterable[Agent], radius: int) -> set[Agent]:
    """Get all agents within a distance radius of the given agents.

    Args:
        agents: Starting agents for the neighborhood.
        radius: Maximum distance to include.

    Returns:
        Set of all agents within the specified radius.
    """
    frontier = agents
    seen = set(frontier)
    for _ in range(radius):
        new_frontier = set()
        for cur in frontier:
            for n in cur.neighbors:
                seen.add(n)
                if n not in seen:
                    new_frontier.add(n)

        frontier = new_frontier
    return seen


def grouped(components: Iterable[Component]) -> dict[Component, list[Component]]:
    """Group components by isomorphism.

    Args:
        components: Components to group.

    Returns:
        Dictionary mapping representative components to lists of isomorphic components.
    """
    grouped: dict[Component, list[Component]] = {}
    for component in components:
        for group in grouped:
            if component.isomorphic(group):
                grouped[group].append(component)
                break
        else:
            grouped[component] = [component]
    return grouped
