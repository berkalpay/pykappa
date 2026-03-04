import math
import colorsys
from typing import TYPE_CHECKING

from graphviz import Source

if TYPE_CHECKING:
    from pykappa.pattern import Component


class ComponentPlot:
    """Stable visualization of a Component across simulation steps."""

    def __init__(self, component: "Component"):
        self.component = component
        self._positions: dict[int, tuple[float, float]] = {}

    def _compute_positions(self) -> dict[int, tuple[float, float]]:
        """Assign each agent a fixed position based on identity, computed once."""
        new_agents = [a for a in self.component.agents if id(a) not in self._positions]

        if new_agents:
            # Place new agents on a sunflower spiral (evenly distributed, deterministic)
            n = len(self._positions)
            golden_angle = math.pi * (3 - math.sqrt(5))
            for i, agent in enumerate(new_agents):
                k = n + i
                r = math.sqrt(k + 1)
                angle = k * golden_angle
                self._positions[id(agent)] = (r * math.cos(angle), r * math.sin(angle))

        return {id(a): self._positions[id(a)] for a in self.component.agents}

    def __call__(self, legend: bool = True):
        agent_types = list(dict.fromkeys(a.type for a in self.component.agents))
        type_color = {
            t: "#{:02x}{:02x}{:02x}".format(
                *[
                    int(c * 255)
                    for c in colorsys.hls_to_rgb(i / len(agent_types), 0.4, 0.8)
                ]
            )
            for i, t in enumerate(agent_types)
        }

        edges = set()
        for a in self.component.agents:
            for b in a.neighbors:
                if a is b:
                    continue
                edges.add(tuple(sorted((id(a), id(b)))))

        pos = self._compute_positions()

        lines = [
            "graph {",
            "  graph [overlap=false];",
            '  node  [shape=circle, width=0.05, height=0.05, fixedsize=true, label="", style=filled];',
            "  edge  [penwidth=0.3];",
        ]
        if legend:
            lines += [
                "  subgraph cluster_legend {",
                "    style=invis;",
                "    node [shape=circle, width=0.15, height=0.15, fixedsize=true, style=filled, fontsize=8, labelloc=r];",
            ]
            for t, color in type_color.items():
                lines.append(f'    legend_{t} [label="{t}", fillcolor="{color}"];')
            lines.append("  }")
        for a in self.component.agents:
            color = type_color[a.type]
            x, y = pos[id(a)]
            lines.append(f'  a{id(a)} [fillcolor="{color}", pos="{x:.3f},{y:.3f}!"];')
        for u, v in edges:
            lines.append(f"  a{u} -- a{v};")
        lines.append("}")

        return Source("\n".join(lines), engine="neato")
