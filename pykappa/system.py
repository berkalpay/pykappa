import os
import shutil
import tempfile
import random
import warnings
import csv
import subprocess
from collections import defaultdict
from functools import cached_property
from typing import Optional, Iterable, Self
from graphviz import Source

from pykappa.mixture import Mixture
from pykappa.rule import Rule
from pykappa.pattern import Component, Pattern, Site
from pykappa.analysis import Monitor
from pykappa._expression import Expression
from pykappa._utils import str_table


class System:
    """A Kappa system containing agents, rules, observables, and variables for simulation."""

    mixture: Mixture  #: The current state of agents and their connections
    rules: dict[str, Rule]  #: Maps rule names to Rule objects
    observables: dict[str, Expression]  #: Maps observable names to expressions
    variables: dict[str, Expression]  #: Maps variable names to expressions
    site_defaults: dict[str, dict[str, str]]  #: Maps agent types to site default states
    tokens: dict[str, float]  #: Maps token names to their current values
    monitor: Optional["Monitor"]  #: Optionally tracks simulation history
    time: float  #: Current simulation time
    tallies: defaultdict[str, dict[str, int]]  #: Tracks rule applications
    _rng: random.Random  # Random number generator for reproducibility of updates

    @classmethod
    def read_ka(cls, filepath: str, seed: Optional[int] = None) -> Self:
        """Read and parse a Kappa .ka file to create a System.

        Args:
            filepath: Path to the Kappa file.
            seed: Random seed for reproducibility.
        """
        with open(filepath) as f:
            return cls.from_ka(f.read(), seed=seed)

    @classmethod
    def from_ka(cls, ka_str: str, seed: Optional[int] = None) -> Self:
        """Create a System from a Kappa (.ka style) string.

        Args:
            ka_str: Kappa language string containing a system definition.
            seed: Random seed for reproducibility.
        """
        from pykappa._parsing import (
            kappa_parser,
            KappaTransformer,
            ExpressionTransformer,
        )

        input_tree = kappa_parser.parse(ka_str)
        assert input_tree.data == "kappa_input"

        variables: dict[str, Expression] = {}
        observables: dict[str, Expression] = {}
        token_inits: list[tuple[str, Expression]] = []
        rules: list[Rule] = []
        inits: list[tuple[Expression, Pattern]] = []

        for child in input_tree.children:
            tag = child.data

            if tag in ["f_rule", "fr_rule", "ambi_rule", "ambi_fr_rule"]:
                new_rules = KappaTransformer().transform(child)
                rules.extend(new_rules)

            elif tag == "variable_declaration":
                name_tree = child.children[0]
                assert name_tree.data == "declared_variable_name"
                name = name_tree.children[0].value.strip("'\"")

                expr_tree = child.children[1]
                assert expr_tree.data == "algebraic_expression"
                value = ExpressionTransformer.from_tree(expr_tree)

                variables[name] = value

            elif tag == "plot_declaration":
                raise NotImplementedError

            elif tag == "observable_declaration":
                label_tree = child.children[0]
                assert isinstance(label_tree, str)
                name = label_tree.strip("'\"")

                expr_tree = child.children[1]
                assert expr_tree.data == "algebraic_expression"
                value = ExpressionTransformer.from_tree(expr_tree)

                observables[name] = value

            elif tag == "signature_declaration":
                pass  # ignore agent signatures

            elif tag == "declared_token":
                pass  # token declarations are handled via init_declaration

            elif tag == "init_declaration":
                amount = ExpressionTransformer.from_tree(child.children[0])
                target = child.children[1]
                if target.data == "declared_token_name":
                    token_inits.append((str(target.children[0]), amount))
                else:
                    pattern = KappaTransformer().transform(target)
                    inits.append((amount, pattern))

            elif tag == "definition":
                pass  # %def: directives not used

            elif tag == "pattern":
                raise NotImplementedError

            else:
                raise TypeError(f"Unsupported input type: {tag}")

        system = cls(None, rules, observables, variables, seed=seed)
        for init in inits:
            system.add(init[1], int(init[0].evaluate(system)))
        for token_name, amount_expr in token_inits:
            system.tokens[token_name] = float(amount_expr.evaluate(system))
        return system

    @classmethod
    def from_kappa(
        cls,
        mixture: Optional[dict[str, int]] = None,
        rules: Optional[Iterable[str]] = None,
        observables: Optional[list[str] | dict[str, str]] = None,
        variables: Optional[dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> Self:
        """Create a System from Kappa strings.

        Args:
            mixture: Dictionary mapping agent patterns to initial counts.
            rules: Iterable of rule strings in Kappa format.
            observables: List of observable expressions or dict mapping names to expressions.
            variables: Dictionary mapping variable names to expressions.
            *args: Additional arguments passed to System constructor.
            **kwargs: Additional keyword arguments passed to System constructor.
        """
        real_rules = []
        if rules is not None:
            for rule in rules:
                real_rules.extend(Rule.list_from_kappa(rule))

        if observables is None:
            real_observables = {}
        elif isinstance(observables, list):
            real_observables = {
                f"o{i}": Expression.from_kappa(obs) for i, obs in enumerate(observables)
            }
        else:
            real_observables = {
                name: Expression.from_kappa(obs) for name, obs in observables.items()
            }

        real_variables = (
            {}
            if variables is None
            else {name: Expression.from_kappa(var) for name, var in variables.items()}
        )

        system = cls(
            None, real_rules, real_observables, real_variables, *args, **kwargs
        )
        if mixture is not None:
            for pattern_str, count in mixture.items():
                system.add(pattern_str, count)
        return system

    def __init__(
        self,
        mixture: Optional[Mixture] = None,
        rules: Optional[Iterable[Rule]] = None,
        observables: Optional[dict[str, Expression]] = None,
        variables: Optional[dict[str, Expression]] = None,
        tokens: Optional[dict[str, float]] = None,
        site_defaults: Optional[dict[str, dict[str, str]]] = None,
        monitor: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            mixture: Initial mixture state.
            rules: Collection of rules to apply.
            observables: Dictionary of observable expressions.
            variables: Dictionary of variable expressions.
            tokens: Dictionary of token names to initial values.
            site_defaults: Maps agent types to site default states.
            monitor: Whether to enable monitoring of simulation history.
            seed: Random seed for reproducibility.
        """
        self._rng = random.Random() if seed is None else random.Random(seed)

        self.rules = (
            {} if rules is None else {f"r{i}": rule for i, rule in enumerate(rules)}
        )

        mixture = Mixture() if mixture is None else mixture
        if any(rule.component_constraint != "any" for rule in self.rules.values()):
            mixture.enable_component_tracking()

        self.observables = {} if observables is None else observables
        self.variables = {} if variables is None else variables
        self.site_defaults = {} if site_defaults is None else dict(site_defaults)

        self._set_mixture(mixture)
        self.time = 0

        self.tokens = {} if tokens is None else dict(tokens)

        self.tallies = defaultdict(lambda: {"applied": 0, "failed": 0})
        self.monitor = Monitor(self) if monitor else None

    def __str__(self):
        return self.kappa_str

    def __getitem__(self, name: str) -> int | float:
        """Get the value of an observable or variable.

        Raises:
            KeyError: If name doesn't correspond to any observable or variable.
        """
        if name in self.observables:
            return self.observables[name].evaluate(self)
        elif name in self.variables:
            return self.variables[name].evaluate(self)
        else:
            raise KeyError(
                f"Name {name} doesn't correspond to a declared observable or variable"
            )

    def __setitem__(self, name: str, value: float) -> None:
        """Update an existing variable to a new numeric value.

        Args:
            name: Name of a declared variable.
            value: New numeric value.

        Raises:
            KeyError: If the name is not a declared variable.
            ValueError: If the declared variable is not a numeric literal.
        """
        if name not in self.variables:
            raise KeyError(f"'{name}' is not a declared variable")
        if self.variables[name].type != "literal":
            raise ValueError(
                f"'{name}' is not a numeric literal and cannot be reassigned"
            )
        self.variables[name] = Expression("literal", value=value)

    @cached_property
    def signatures(self) -> dict[str, frozenset[str]]:
        """The complete site interface for each agent type inferrred from all rules."""
        sites_by_type: dict[str, set[str]] = defaultdict(set)
        for rule in self.rules.values():
            for pattern in (rule.left, rule.right):
                for agent in pattern.agents:
                    if agent is not None:
                        sites_by_type[agent.type].update(site.label for site in agent)
        return {
            agent_type: frozenset(sites) for agent_type, sites in sites_by_type.items()
        }

    @property
    def tallies_str(self) -> str:
        """A formatted string showing how many times each rule has been applied."""
        return str_table(
            [
                [str(rule), tallies["applied"], tallies["failed"]]
                for rule, tallies in self.tallies.items()
            ],
            header=["Rule", "Applied", "Failed"],
        )

    @cached_property
    def _reversible_rules(self) -> list[tuple[str, str]]:
        """Find forward/reverse rule pairs by checking pattern symmetry."""
        names = list(self.rules.keys())
        pairs = []
        used = set()

        for i, name_a in enumerate(names):
            if name_a in used:
                continue
            rule_a = self.rules[name_a]

            for name_b in names[i + 1 :]:
                if name_b in used:
                    continue
                rule_b = self.rules[name_b]

                if rule_a.left.n_isomorphisms(
                    rule_b.right
                ) and rule_a.right.n_isomorphisms(rule_b.left):
                    pairs.append((name_a, name_b))
                    used.add(name_a)
                    used.add(name_b)
                    break

        return pairs

    @property
    def kappa_str(self) -> str:
        """The system representation in Kappa (.ka style) format."""

        kappa_list = []

        # Append the inferred agent signature at the top
        for agent, sites in self.signatures.items():
            sig = ", ".join(sites)
            kappa_list.append(f"%agent: {agent}({sig})")

        # Format reversible rules with <-> notation
        pairs = self._reversible_rules
        paired = {name for pair in pairs for name in pair}
        for fwd_name, rev_name in pairs:
            fwd = self.rules[fwd_name]
            rev = self.rules[rev_name]
            kappa_list.append(
                f"{fwd.left.kappa_str} <-> {fwd.right.kappa_str} "
                f"@ {fwd._rate_str}, {rev._rate_str}"
            )
        # Otherwise format with -> notation
        for name, rule in self.rules.items():
            if name not in paired:
                kappa_list.append(rule.kappa_str)

        for var_name, var in self.variables.items():
            kappa_list.append(f"%var: '{var_name}' {var.kappa_str}")

        for obs_name, obs in self.observables.items():
            obs_str = (
                f"|{obs.kappa_str}|" if isinstance(obs, Component) else obs.kappa_str
            )
            kappa_list.append(f"%obs: '{obs_name}' {obs_str}")

        kappa_list.append(self.mixture.kappa_str)

        return "\n".join(kappa_list)

    def to_ka(self, filepath: str) -> None:
        """Write system information to a Kappa file."""
        with open(filepath, "w") as f:
            f.write(self.kappa_str)

    def _set_mixture(self, mixture: Mixture) -> None:
        """Set the system's mixture and update tracking."""
        self.mixture = mixture
        for rule in self.rules.values():
            for component in rule.left.components:
                if component not in mixture._embeddings:
                    mixture._track_component(component)
        for expr in [*self.observables.values(), *self.variables.values()]:
            for component_expr in expr.filter("component_pattern"):
                mixture._track_component(component_expr.attrs["value"])

    def _enforce_signature(self, agent: "Agent") -> None:
        """Validate agent type and sites against the inferred signature and fill missing sites.

        Raises:
            ValueError: If the agent type is unknown or has sites not in the signature.
        """
        if not self.signatures:
            return
        known = self.signatures.get(agent.type)
        if known is None:
            raise ValueError(
                f"Agent type '{agent.type}' is not declared by any rule. "
                f"Known agent types: {set(self.signatures)}"
            )
        unknown = {s.label for s in agent} - known
        if unknown:
            raise ValueError(
                f"Agent '{agent.type}' has unknown site(s) {unknown}. "
                f"Known sites for this type: {known}"
            )
        for label in known - agent.interface.keys():
            agent.interface[label] = site = Site(
                label, self.site_defaults.get(agent.type, {}).get(label, "?"), "."
            )
            site.agent = agent

    def add(self, pattern: Pattern | Component | str, n_copies: int = 1) -> None:
        """Add instances of a pattern or component to the mixture using inferred agent signatures."""
        if isinstance(pattern, str):
            pattern = Pattern.from_kappa(pattern)
        components = [pattern] if isinstance(pattern, Component) else pattern.components
        for component in components:
            agent_map = {agent: agent.detached() for agent in component.agents}
            for agent in component.agents:
                for site in agent:
                    if site.coupled:
                        agent_map[agent][site.label].partner = agent_map[
                            site.partner.agent
                        ][site.partner.label]
            copied = Component(list(agent_map.values()))
            for agent in copied.agents:
                self._enforce_signature(agent)
            self.mixture.add(copied, n_copies)

    @property
    def reactivity(self) -> float:
        """The total reactivity of the system."""
        return sum(rule.reactivity(self) for rule in self.rules.values())

    def update(self) -> None:
        """Perform one simulation step."""

        if self.monitor is not None and not self.monitor.history["time"]:
            self.monitor.update()

        if (reactivity := self.reactivity) == 0:
            warnings.warn("system has no reactivity", RuntimeWarning)
            if self.monitor is not None:
                self.monitor.update()
            return

        self.time += self._rng.expovariate(reactivity)

        rule = self._rng.choices(
            list(self.rules.values()),
            weights=[rule.reactivity(self) for rule in self.rules.values()],
        )[0]

        # Apply the rule
        update = rule._select(self.mixture)
        if update is not None:
            self.tallies[str(rule)]["applied"] += 1
            for agent in update.agents_to_add:
                self._enforce_signature(agent)
            self.mixture._apply_update(update)
            for expr, name in rule.token_updates:
                self.tokens[name] += expr.evaluate(self)
        else:
            self.tallies[str(rule)]["failed"] += 1

        if self.monitor is not None:
            self.monitor.update()

    def apply(self, transformation: str, n: int = 1) -> None:
        """Apply a transformation immediately for a specified number of times.

        Unlike `update`, this does not advance simulation time or use stochastic
        selection — the rule fires exactly ``n`` times using randomly chosen
        embeddings.

        Args:
            transformation: Kappa string representation of the rule.
            n: Number of times to apply the rule.
        """
        rule = Rule.from_kappa(transformation + " @ 0")
        for _ in range(n):
            update = Rule._select(rule, self.mixture)
            if update is not None:
                for agent in update.agents_to_add:
                    self._enforce_signature(agent)
                self.mixture._apply_update(update)

    def update_via_kasim(self, time: float) -> None:
        """Simulate for a given amount of time using KaSim.

        Note:
            KaSim must be installed and in the PATH.
            Some features are not compatible between PyKappa and KaSim.
        """
        assert shutil.which("KaSim"), "KaSim not found in the PATH."

        if any(rule.n_symmetries > 1 for rule in self.rules.values()):
            warnings.warn(
                "Some rules have multiple symmetries. "
                "PyKappa normalizes reactivities accordingly: results may differ from KaSim."
            )

        history = None  # the observable history

        with tempfile.TemporaryDirectory() as tmpdirname:
            snap_path = os.path.join(tmpdirname, "snap.ka")
            out_path = os.path.join(tmpdirname, "out.ka")
            in_path = os.path.join(tmpdirname, "in.ka")

            output_lines = [
                self.kappa_str,
                f'%mod: alarm {time} do $SNAPSHOT "{snap_path}";',
            ]
            if self.observables:
                output_lines.append("%mod: [true] do $PLOTENTRY; repeat [true]")

            # Run KaSim
            with open(in_path, "w") as f:
                f.write("\n".join(output_lines))

            subprocess.run(
                ["KaSim", in_path, "-l", str(time), "-d", tmpdirname, "-o", out_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Read KaSim output
            with open(snap_path) as f:
                content = f.read()

            if self.observables:
                with open(out_path) as f:
                    reader = csv.reader(f)
                    header = next(row for row in reader if row and row[0] == "[T]")
                    columns = ["time", *header[1:]]
                    history = {name: [] for name in columns}

                    for row in reader:
                        history["time"].append(self.time + float(row[0]))
                        for name, value in zip(columns[1:], row[1:]):
                            history[name].append(float(value))

        content = content.replace(
            ",\n", ", "
        )  # KaSim splits long components across lines
        output_kappa_str = "".join(
            line.split("/")[0] + line.split("/")[-1]
            for line in content.splitlines(keepends=True)
            if line.startswith("%init")
        )

        # Apply the update
        self._set_mixture(System.from_ka(output_kappa_str).mixture)
        self.time += time

        # Update the monitor
        if self.monitor is not None and history is not None:
            for name, values in history.items():
                self.monitor.history[name].extend(values)

    def kd_table(self, volume: float = 1.0) -> str:
        """Summarize kinetic constants of two-component binding/unbinding rules
        given volume in liters."""
        from pykappa.analysis import _kd_table

        return _kd_table(self, volume=volume)

    def rule_graph(self) -> Source:
        """Visualize a ruleset as a site graph of local transformations.

        Solid edges = bond formation; dashed edges = bond breaking. Sites that
        change state show their transition as ``site {old→new}``. Creation and
        degradation are shown as directed edges to/from a sink node.

        Note:
            This is a lossy projection that neglects conditions of transformations;
            multiple rulesets can yield the same graph.
        """
        from pykappa.analysis import _rule_graph

        return _rule_graph(self)

    def contact_map(self) -> Source:
        """Generate a graphviz contact map using the KaSa static analyzer."""
        from pykappa.analysis import _contact_map

        return _contact_map(self)
