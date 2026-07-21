from dataclasses import dataclass, field
from typing import Optional, Iterable, Iterator, Self
from contextlib import contextmanager

from pykappa.pattern import Site, Agent, Component, Pattern, Embedding
from pykappa._utils import IndexedSet


@dataclass(frozen=True)
class _Edge:
    """Represents bonds between sites. Edge(x, y) equals Edge(y, x)."""

    site1: Site
    site2: Site

    def __eq__(self, other):
        return (self.site1 == other.site1 and self.site2 == other.site2) or (
            self.site1 == other.site2 and self.site2 == other.site1
        )

    def __hash__(self):
        return hash(frozenset((self.site1, self.site2)))


class Mixture:
    """A collection of agents and their connections.

    Optionally tracks connected components.
    """

    agents: IndexedSet[Agent]
    _components: Optional[IndexedSet[Component]]  # Components if tracking is enabled
    _embeddings: dict[Component, IndexedSet[Embedding]]  # Cache of embeddings
    _max_embedding_width: int  # Max diameter, to compute re-embedding neighborhoods

    @classmethod
    def from_kappa(cls, patterns: dict[str, int]) -> Self:
        """Create a mixture from Kappa pattern strings and counts.

        Args:
            patterns: Dictionary mapping pattern strings to copy counts.
        """
        real_patterns = []
        for pattern, count in patterns.items():
            real_patterns.extend([Pattern.from_kappa(pattern)] * count)
        return cls(real_patterns)

    def __init__(
        self,
        patterns: Optional[Iterable[Pattern]] = None,
        track_components: bool = False,
    ):
        self.agents = IndexedSet()
        self.agents.create_index("type", lambda a: [a.type])
        self._components = IndexedSet() if track_components else None
        if self._components is not None:
            self._components.create_index("agent", lambda c: c.agents)
        self._embeddings = {}
        self._max_embedding_width = 0

        if patterns is not None:
            for pattern in patterns:
                self.add(pattern)

    def __iter__(self) -> Iterator[Component]:
        yield from self.components

    def __str__(self):
        return self.kappa_str

    @property
    def kappa_str(self) -> str:
        """The mixture in Kappa format with %init declarations."""

        # Group components by isomorphism
        grouped: dict[Component, list[Component]] = {}
        for component in self:
            for group in grouped:
                if component.isomorphic(group):
                    grouped[group].append(component)
                    break
            else:
                grouped[component] = [component]

        return "\n".join(
            f"%init: {len(components)} {group.kappa_str}"
            for group, components in grouped.items()
        )

    @property
    def component_tracking(self) -> bool:
        """Whether connected components are being tracked."""
        return self._components is not None

    @property
    def components(self) -> IndexedSet[Component]:
        if self.component_tracking:  # Use cached components if tracking
            return self._components

        components = IndexedSet()
        unassigned = set(self.agents)
        while unassigned:
            component_agents = set(next(iter(unassigned))._depth_first_traversal)
            component_agents.intersection_update(self.agents)
            components.add(Component(component_agents))
            unassigned.difference_update(component_agents)
        return components

    def add(
        self,
        pattern: Pattern | Component | str,
        n_copies: int = 1,
    ) -> None:
        """Add instances of a pattern or component to the mixture.

        Raises:
            AssertionError: If pattern is underspecified.
        """
        if isinstance(pattern, Component):
            for _ in range(n_copies):
                self._add_component(pattern)
            return

        if isinstance(pattern, str):
            pattern = Pattern.from_kappa(pattern)

        assert pattern.instantiable, "Pattern isn't specific enough to instantiate."
        for _ in range(n_copies):
            for component in pattern.components:
                self._add_component(component)

    def _add_component(self, component: Component) -> None:
        component_ordered = list(component.agents)
        new_agents = [agent._detached() for agent in component_ordered]
        new_edges = set()

        # Reconstruct the bond structure in the copied agents
        for i, agent in enumerate(component_ordered):
            for site in agent:
                if site.coupled:
                    partner = site.partner
                    i_partner = component_ordered.index(partner.agent)
                    new_site = new_agents[i][site.label]
                    new_partner = new_agents[i_partner][partner.label]
                    new_edges.add(_Edge(new_site, new_partner))

        update = _MixtureUpdate(agents_to_add=set(new_agents), edges_to_add=new_edges)
        self._apply_update(update)

    def remove(self, component: Component) -> None:
        """Remove a component from the mixture."""
        update = _MixtureUpdate()
        for agent in component:
            update.remove_agent(agent)
        self._apply_update(update)

    def embeddings(self, component: Component) -> IndexedSet[Embedding]:
        """Get embeddings of a tracked component (not accounting for symmetries).

        Raises:
            KeyError: If component is not being tracked.
        """
        try:
            return self._embeddings[component]
        except KeyError as e:
            e.add_note(
                f"Undeclared component: {component}. To track it, add it as an observable."
            )
            raise

    def embeddings_in_component(
        self, match_pattern: Component, mixture_component: Component
    ) -> IndexedSet[Embedding]:
        """Get embeddings of a pattern within a specific component."""
        if not self.component_tracking:
            raise RuntimeError("Component tracking is not enabled.")
        return self._embeddings[match_pattern].lookup("component", mixture_component)

    def _track_component(self, component: Component):
        """Start tracking embeddings of a component."""
        self._max_embedding_width = max(component.diameter, self._max_embedding_width)
        embeddings = IndexedSet(component.embeddings(self))
        embeddings.create_index("agent", lambda e: iter(e.values()))
        self._embeddings[component] = embeddings

        if self.component_tracking:
            embeddings.create_index(
                "component",
                lambda e: [self.components.lookup_one("agent", next(iter(e.values())))],
            )

    def _apply_update(self, update: "_MixtureUpdate") -> None:
        """Apply a collection of changes to the mixture."""
        # Clear embeddings involving agents that will change
        for agent in update.touched_before:
            for tracked in self._embeddings:
                self._embeddings[tracked].remove_by("agent", agent)

        # Modify the graph structure
        for edge in update.edges_to_remove:
            self._remove_edge(edge)
        for agent in update.agents_to_remove:
            self._remove_agent(agent)
        for agent in update.agents_to_add:
            self._add_agent(agent)
        for edge in update.edges_to_add:
            self._add_edge(edge)

        # Re-embed tracked components in the updated region around modified agents
        update_region = Agent.neighborhood(
            update.touched_after, self._max_embedding_width
        )
        update_region = IndexedSet(update_region)
        update_region.create_index("type", lambda a: [a.type])
        for component_pattern in self._embeddings:
            new_embeddings = component_pattern.embeddings(update_region)
            for e in new_embeddings:
                self._embeddings[component_pattern].add(e)

    def _add_agent(self, agent: Agent) -> None:
        """Add an agent to the mixture (should not have any bound sites)."""
        assert all(site.partner == "." for site in agent)
        assert agent.instantiable
        self.agents.add(agent)

        if self.component_tracking:
            self.components.add(Component([agent]))

    def _remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the mixture (bonds must be removed first)."""
        assert all(site.partner == "." for site in agent)
        self.agents.remove(agent)

        if self.component_tracking:
            component = self.components.lookup_one("agent", agent)
            self.components.remove(component)

    def _add_edge(self, edge: _Edge) -> None:
        """Add a bond between two sites."""
        assert edge.site1.agent in self.agents
        assert edge.site2.agent in self.agents
        edge.site1.partner = edge.site2
        edge.site2.partner = edge.site1

        if not self.component_tracking:
            return

        # Check if the edge merges two components
        component1 = self.components.lookup_one("agent", edge.site1.agent)
        component2 = self.components.lookup_one("agent", edge.site2.agent)
        if component1 == component2:
            return

        # Merge smaller component into larger for efficiency
        if len(component2) > len(component1):
            component1, component2 = component2, component1
        with self._relocate_embeddings(component2):
            self.components.remove(component2)
            for agent in component2:
                component1.agents.add(agent)
                self.components.indices["agent"][agent] = [component1]

    def _remove_edge(self, edge: _Edge) -> None:
        """Remove a bond between two sites."""
        assert edge.site1.partner == edge.site2
        assert edge.site2.partner == edge.site1
        edge.site1.partner = "."
        edge.site2.partner = "."

        if not self.component_tracking:
            return

        agent1: Agent = edge.site1.agent
        agent2: Agent = edge.site2.agent
        old_component = self.components.lookup_one("agent", agent1)
        assert old_component == self.components.lookup_one("agent", agent2)

        # Check if edge removal splits the component
        maybe_new_component = Component(agent1._depth_first_traversal)
        if agent2 in maybe_new_component:
            return

        # Handle the split
        new_component1 = maybe_new_component
        new_component2 = Component(agent2._depth_first_traversal)
        with self._relocate_embeddings(old_component):
            self.components.remove(old_component)
            self.components.add(new_component1)
            self.components.add(new_component2)

    @contextmanager
    def _relocate_embeddings(self, component: Component):
        """Temporarily evacuate and restore embeddings during component restructuring."""
        relocated = {}
        # Save and remove embeddings that reference the restructured component
        for tracked in self._embeddings:
            relocated[tracked] = list(
                self._embeddings[tracked].lookup("component", component)
            )
            for e in relocated[tracked]:
                self._embeddings[tracked].remove(e)

        try:
            yield
        finally:
            # Restore embeddings after restructuring
            for tracked in self._embeddings:
                for e in relocated.get(tracked, []):
                    self._embeddings[tracked].add(e)


@dataclass
class _MixtureUpdate:
    """Specifies changes to be applied to a mixture."""

    agents_to_add: set[Agent] = field(default_factory=set)
    agents_to_remove: set[Agent] = field(default_factory=set)
    edges_to_add: set[_Edge] = field(default_factory=set)
    edges_to_remove: set[_Edge] = field(default_factory=set)
    agents_changed: set[Agent] = field(default_factory=set)  # Internal state changes

    def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent based on a template (sites will be emptied)."""
        new_agent = agent._detached()
        self.agents_to_add.add(new_agent)
        return new_agent

    def remove_agent(self, agent: Agent) -> None:
        """Specify to remove an agent and its edges from the mixture."""
        self.agents_to_remove.add(agent)
        for site in agent:
            if site.coupled:
                self.edges_to_remove.add(_Edge(site, site.partner))

    def connect_sites(self, site1: Site, site2: Site) -> None:
        """Specify to create an edge between two sites. If the sites
        are bound to other sites, indicates to remove those edges.
        """
        if site1.coupled and site1.partner != site2:
            self.disconnect_site(site1)
        if site2.coupled and site2.partner != site1:
            self.disconnect_site(site2)
        if not site1.partner == site2:
            self.edges_to_add.add(_Edge(site1, site2))

    def disconnect_site(self, site: Site) -> None:
        """Specify that a site should be unbound."""
        if site.coupled:
            self.edges_to_remove.add(_Edge(site, site.partner))

    @property
    def touched_before(self) -> set[Agent]:
        """The agents that will be changed or removed by this update."""
        touched = self.agents_changed | set(self.agents_to_remove)

        for edge in self.edges_to_remove:
            touched.add(edge.site1.agent)
            touched.add(edge.site2.agent)

        for edge in self.edges_to_add:
            a, b = edge.site1.agent, edge.site2.agent
            if a not in self.agents_to_add:
                touched.add(a)
            if b not in self.agents_to_add:
                touched.add(b)

        return touched

    @property
    def touched_after(self) -> set[Agent]:
        """The agents that will be changed or added after this update."""
        touched = self.agents_changed | set(self.agents_to_add)

        for edge in self.edges_to_add:
            touched.add(edge.site1.agent)
            touched.add(edge.site2.agent)

        for edge in self.edges_to_remove:
            a, b = edge.site1.agent, edge.site2.agent
            if a not in self.agents_to_remove:
                touched.add(a)
            if b not in self.agents_to_remove:
                touched.add(b)

        return touched
