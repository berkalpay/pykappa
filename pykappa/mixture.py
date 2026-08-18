"""Manages collections of agents."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Iterable, Iterator, Self
from contextlib import contextmanager

from pykappa.pattern import Site, Agent, Component, Pattern, Embedding
from pykappa._utils import IndexedSet, IndexedSetView, OrderedSet


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


class _Connectivity(set[_Edge]):
    """Tracks cycle-closing bonds in an implicit spanning forest."""

    @staticmethod
    def _other(edge: _Edge, agent: Agent) -> Agent:
        return edge.site2.agent if edge.site1.agent == agent else edge.site1.agent

    @staticmethod
    def _edges(agent: Agent) -> Iterable[_Edge]:
        return (_Edge(site, site.partner) for site in agent if site._coupled)

    def _tree_side(self, start: Agent, limit: int | None = None) -> set[Agent]:
        """Return a tree-connected side, stopping early once ``limit`` is exceeded."""
        side: set[Agent] = set()
        stack = [start]
        while stack:
            agent = stack.pop()
            if agent in side:
                continue
            side.add(agent)
            if limit is not None and len(side) > limit:
                return side
            stack.extend(
                self._other(edge, agent)
                for edge in self._edges(agent)
                if edge not in self
            )
        return side

    def smaller_tree_side(
        self, agent1: Agent, agent2: Agent, component_size: int
    ) -> set[Agent]:
        """Return the smaller side of a forest bond removed from the graph."""
        limit = component_size // 2
        first = self._tree_side(agent1, limit)
        return first if len(first) <= limit else self._tree_side(agent2)

    def replacement_edge(self, side: set[Agent]) -> _Edge | None:
        """Find an extra bond crossing from ``side`` to the other tree side."""
        for agent in side:
            for edge in self._edges(agent):
                if edge in self and self._other(edge, agent) not in side:
                    return edge
        return None


class Mixture:
    """A collection of agents and their connections.

    Optionally tracks connected components.
    """

    _agents: IndexedSet[Agent]
    _components: Optional[IndexedSet[Component]]  # Components if tracking is enabled
    _embeddings: dict[Component, IndexedSet[Embedding]]  # Cache of embeddings
    _connectivity: Optional[_Connectivity]

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
        self._agents = IndexedSet()
        self._agents.create_index("type", lambda a: [a.type])
        self._components = IndexedSet() if track_components else None
        if self._components is not None:
            self._components.create_index("agent", lambda c: c.agents)
        self._embeddings = {}
        self._connectivity = _Connectivity() if track_components else None

        if patterns is not None:
            for pattern in patterns:
                self._add(pattern)

    def __iter__(self) -> Iterator[Agent]:
        yield from self.agents

    def __len__(self) -> int:
        return len(self.agents)

    def __str__(self):
        return self.kappa_str

    @property
    def agents(self) -> IndexedSetView[Agent]:
        """The agents in the mixture."""
        return self._agents.view

    @property
    def kappa_str(self) -> str:
        """The mixture in Kappa format with %init declarations."""

        # Group components by isomorphism
        grouped: dict[Component, list[Component]] = {}
        for component in self.components:
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
    def kappa_str_with_agent_ids(self) -> str:
        """Kappa representation with IDs and one `%init: 1` per component."""
        return "\n".join(
            f"%init: 1 {component.kappa_str_with_agent_ids}"
            for component in self.components
        )

    @property
    def component_tracking(self) -> bool:
        """Whether connected components are being tracked."""
        return self._components is not None

    @property
    def components(self) -> IndexedSetView[Component]:
        if self.component_tracking:  # Use cached components if tracking
            return self._components.view

        components = IndexedSet()
        unassigned = set(self.agents)
        for agent in self.agents:
            if agent not in unassigned:
                continue
            component_agents = [
                member
                for member in agent._depth_first_traversal
                if member in unassigned
            ]
            components.add(Component(component_agents))
            unassigned.difference_update(component_agents)
        return components.view

    def _add(self, pattern: Pattern | Component | str, n_copies: int = 1) -> None:
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

        assert pattern._instantiable, "Pattern isn't specific enough to instantiate."
        for _ in range(n_copies):
            for component in pattern.components:
                self._add_component(component)

    def _add_component(
        self,
        component: Component,
        prepare_agent: Callable[[Agent], None] | None = None,
    ) -> None:
        """Copy and add a component."""
        component_ordered = list(component.agents)
        new_agents = [agent._detached() for agent in component_ordered]
        new_edges = OrderedSet()
        copied_agents = dict(zip(component_ordered, new_agents, strict=True))

        if prepare_agent is not None:
            for agent in new_agents:
                prepare_agent(agent)

        # Reconstruct the bond structure in the copied agents
        for i, agent in enumerate(component_ordered):
            for site in agent:
                if site._coupled:
                    partner = site.partner
                    new_site = new_agents[i][site.label]
                    new_partner = copied_agents[partner.agent][partner.label]
                    new_edges.add(_Edge(new_site, new_partner))

        update = _MixtureUpdate(
            agents_to_add=OrderedSet(new_agents), edges_to_add=new_edges
        )
        self._apply_update(update)

    def _remove_component(self, component: Component) -> None:
        """Remove a component from the mixture."""
        update = _MixtureUpdate()
        for agent in component:
            update.remove_agent(agent)
        self._apply_update(update)

    def embeddings(self, component: Component) -> IndexedSetView[Embedding]:
        """Get embeddings of a tracked component (not accounting for symmetries).

        Raises:
            KeyError: If component is not being tracked.
        """
        try:
            return self._embeddings[component].view
        except KeyError as e:
            e.add_note(
                f"Undeclared component: {component}. To track it, add it as an observable."
            )
            raise

    def embeddings_in_component(
        self, match_pattern: Component, mixture_component: Component
    ) -> IndexedSetView[Embedding]:
        """Get embeddings of a pattern within a specific component."""
        if not self.component_tracking:
            raise RuntimeError("Component tracking is not enabled.")
        return (
            self._embeddings[match_pattern].lookup("component", mixture_component).view
        )

    def _track_component(self, component: Component):
        """Start tracking embeddings of a component."""
        embeddings = IndexedSet(component.embeddings(self))
        embeddings.create_index("agent", lambda e: iter(e.values()))
        self._embeddings[component] = embeddings

        if self.component_tracking:
            embeddings.create_index(
                "component",
                lambda e: [self.components.lookup_one("agent", next(iter(e.values())))],
            )

    def _components_containing(self, agents: Iterable[Agent]) -> set[Component]:
        """Return tracked components containing any of the given agents."""
        if not self.component_tracking:
            return set()
        return {self.components.lookup_one("agent", agent) for agent in agents}

    def _apply_update(
        self, update: "_MixtureUpdate"
    ) -> tuple[set[Component], set[Component]]:
        """Apply a collection of changes and return the affected components."""
        previous_components = self._components_containing(update.touched_before)
        changes: set[tuple[str, str | None]] = {
            (site.agent.type, site.label) for site in update.sites_changed
        }
        for edge in (*update.edges_to_remove, *update.edges_to_add):
            changes.add((edge.site1.agent.type, edge.site1.label))
            changes.add((edge.site2.agent.type, edge.site2.label))
        changes.update(
            (agent.type, None)
            for agent in (*update.agents_to_remove, *update.agents_to_add)
        )
        affected_patterns = {
            pattern: pattern._embedding_cache_info[0]
            for pattern in self._embeddings
            if changes & pattern._embedding_cache_info[1]
        }

        # Clear embeddings involving agents that will change
        for agent in update.touched_before:
            for tracked in affected_patterns:
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

        # Re-embed each tracked pattern as far as its own diameter requires
        update_regions: dict[int, IndexedSet[Agent]] = {}
        for component_pattern, width in affected_patterns.items():
            if width not in update_regions:
                update_region = IndexedSet(
                    Agent.neighborhood(update.touched_after, width)
                )
                update_region.create_index("type", lambda agent: [agent.type])
                update_regions[width] = update_region
            else:
                update_region = update_regions[width]
            new_embeddings = component_pattern.embeddings(update_region)
            for e in new_embeddings:
                self._embeddings[component_pattern].add(e)

        return previous_components, self._components_containing(update.touched_after)

    def _add_agent(self, agent: Agent) -> None:
        """Add an agent to the mixture (should not have any bound sites)."""
        assert all(site.partner == "." for site in agent)
        assert agent._instantiable
        self._agents.add(agent)

        if self.component_tracking:
            self._components.add(Component([agent]))

    def _remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the mixture (bonds must be removed first)."""
        assert all(site.partner == "." for site in agent)
        self._agents.remove(agent)

        if self.component_tracking:
            component = self.components.lookup_one("agent", agent)
            self._components.remove(component)

    def _add_edge(self, edge: _Edge) -> None:
        """Add a bond between two sites."""
        assert edge.site1.agent in self.agents
        assert edge.site2.agent in self.agents
        edge.site1._set_partner(edge.site2)
        edge.site2._set_partner(edge.site1)

        if not self.component_tracking:
            return

        # Check if the edge merges two components
        component1 = self.components.lookup_one("agent", edge.site1.agent)
        component2 = self.components.lookup_one("agent", edge.site2.agent)
        if component1 == component2:
            self._connectivity.add(edge)
            return

        # Merge smaller component into larger for efficiency
        if len(component2) > len(component1):
            component1, component2 = component2, component1
        with self._relocate_embeddings(component2):
            self._components.remove(component2)
            for agent in component2:
                component1._agents.add(agent)
                self._components.indices["agent"][agent] = [component1]

    def _remove_edge(self, edge: _Edge) -> None:
        """Remove a bond between two sites."""
        assert edge.site1.partner == edge.site2
        assert edge.site2.partner == edge.site1
        edge.site1._set_partner(".")
        edge.site2._set_partner(".")

        if not self.component_tracking:
            return

        agent1: Agent = edge.site1.agent
        agent2: Agent = edge.site2.agent
        old_component = self.components.lookup_one("agent", agent1)
        assert old_component == self.components.lookup_one("agent", agent2)

        if edge in self._connectivity:
            self._connectivity.remove(edge)
            return

        smaller_side = self._connectivity.smaller_tree_side(
            agent1, agent2, len(old_component)
        )
        replacement = self._connectivity.replacement_edge(smaller_side)
        if replacement is not None:
            self._connectivity.remove(replacement)
            return

        # The component is split
        new_component1 = Component(
            agent for agent in old_component if agent in smaller_side
        )
        new_component2 = Component(
            agent for agent in old_component if agent not in smaller_side
        )
        with self._relocate_embeddings(old_component):
            self._components.remove(old_component)
            self._components.add(new_component1)
            self._components.add(new_component2)

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

    agents_to_add: OrderedSet[Agent] = field(default_factory=OrderedSet)
    agents_to_remove: OrderedSet[Agent] = field(default_factory=OrderedSet)
    edges_to_add: OrderedSet[_Edge] = field(default_factory=OrderedSet)
    edges_to_remove: OrderedSet[_Edge] = field(default_factory=OrderedSet)
    agents_changed: OrderedSet[Agent] = field(default_factory=OrderedSet)
    sites_changed: OrderedSet[Site] = field(default_factory=OrderedSet)

    def create_agent(self, agent: Agent) -> Agent:
        """Create a new agent based on a template (sites will be emptied)."""
        new_agent = agent._detached()
        self.agents_to_add.add(new_agent)
        return new_agent

    def remove_agent(self, agent: Agent) -> None:
        """Specify to remove an agent and its edges from the mixture."""
        self.agents_to_remove.add(agent)
        for site in agent:
            if site._coupled:
                self.edges_to_remove.add(_Edge(site, site.partner))

    def set_site_state(self, site: Site, state: str) -> None:
        """Set a site's state and record it for embedding-cache maintenance."""
        if site.state != state:
            self.agents_changed.add(site.agent)
            self.sites_changed.add(site)
            site._set_state(state)

    def connect_sites(self, site1: Site, site2: Site) -> None:
        """Specify to create an edge between two sites. If the sites
        are bound to other sites, indicates to remove those edges.
        """
        if site1._coupled and site1.partner != site2:
            self.disconnect_site(site1)
        if site2._coupled and site2.partner != site1:
            self.disconnect_site(site2)
        if not site1.partner == site2:
            self.edges_to_add.add(_Edge(site1, site2))

    def disconnect_site(self, site: Site) -> None:
        """Specify that a site should be unbound."""
        if site._coupled:
            self.edges_to_remove.add(_Edge(site, site.partner))

    @property
    def touched_before(self) -> OrderedSet[Agent]:
        """The agents that will be changed or removed by this update."""
        touched = OrderedSet(self.agents_changed)
        touched.update(self.agents_to_remove)

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
    def touched_after(self) -> OrderedSet[Agent]:
        """The agents that will be changed or added after this update."""
        touched = OrderedSet(self.agents_changed)
        touched.update(self.agents_to_add)

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
