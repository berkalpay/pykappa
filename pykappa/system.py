import os
import shutil
import tempfile
import random
import warnings
from collections import defaultdict
from functools import cached_property
from typing import Optional, Iterable, Self

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

from pykappa.mixture import Mixture, ComponentMixture
from pykappa.rule import Rule, KappaRule, KappaRuleUnimolecular, KappaRuleBimolecular
from pykappa.pattern import Component, Pattern
from pykappa.algebra import Expression
from pykappa.utils import str_table


class System:
    """A Kappa system containing agents, rules, observables, and variables for simulation.

    Attributes:
        mixture: The current state of agents and their connections.
        rules: Dictionary mapping rule names to Rule objects.
        observables: Dictionary mapping observable names to expressions.
        variables: Dictionary mapping variable names to expressions.
        monitor: Optional Monitor object for tracking simulation history.
        time: Current simulation time.
        tallies: Dictionary tracking rule application counts.
    """

    mixture: Mixture
    rules: dict[str, Rule]
    observables: dict[str, Expression]
    variables: dict[str, Expression]
    monitor: Optional["Monitor"]
    time: float
    tallies: defaultdict[str, dict[str, int]]

    @classmethod
    def read_ka(cls, filepath: str) -> Self:
        """Read and parse a Kappa .ka file to create a System.

        Args:
            filepath: Path to the Kappa file.

        Returns:
            A new System instance parsed from the file.
        """
        with open(filepath) as f:
            return cls.from_ka(f.read())

    @classmethod
    def from_ka(cls, ka_str: str) -> Self:
        """Create a System from a Kappa (.ka style) string.

        Args:
            ka_str: Kappa language string containing a system definition.

        Returns:
            A new System instance parsed from the string.
        """
        from pykappa.grammar import (
            kappa_parser,
            parse_tree_to_expression,
            PatternBuilder,
            RuleBuilder,
        )

        input_tree = kappa_parser.parse(ka_str)
        assert input_tree.data == "kappa_input"

        variables: dict[str, Expression] = {}
        observables: dict[str, Expression] = {}
        rules: list[Rule] = []
        system_params: dict[str, int] = {}
        inits: list[tuple[Expression, Pattern]] = []

        for child in input_tree.children:
            tag = child.data

            if tag in ["f_rule", "fr_rule", "ambi_rule", "ambi_fr_rule"]:
                new_rules = RuleBuilder(child).objects
                rules.extend(new_rules)

            elif tag == "variable_declaration":
                name_tree = child.children[0]
                assert name_tree.data == "declared_variable_name"
                name = name_tree.children[0].value.strip("'\"")

                expr_tree = child.children[1]
                assert expr_tree.data == "algebraic_expression"
                value = parse_tree_to_expression(expr_tree)

                variables[name] = value

            elif tag == "plot_declaration":
                raise NotImplementedError

            elif tag == "observable_declaration":
                label_tree = child.children[0]
                assert isinstance(label_tree, str)
                name = label_tree.strip("'\"")

                expr_tree = child.children[1]
                assert expr_tree.data == "algebraic_expression"
                value = parse_tree_to_expression(expr_tree)

                observables[name] = value

            elif tag == "signature_declaration":
                raise NotImplementedError

            elif tag == "init_declaration":
                expr_tree = child.children[0]
                assert expr_tree.data == "algebraic_expression"
                amount = parse_tree_to_expression(expr_tree)

                pattern_tree = child.children[1]
                if pattern_tree.data == "declared_token_name":
                    raise NotImplementedError
                assert pattern_tree.data == "pattern"
                pattern = PatternBuilder(pattern_tree).object

                inits.append((amount, pattern))

            elif tag == "declared_token":
                raise NotImplementedError

            elif tag == "definition":
                reserved_name_tree = child.children[0]
                assert reserved_name_tree.data == "reserved_name"
                name = reserved_name_tree.children[0].value.strip("'\"")

                value_tree = child.children[1]
                assert value_tree.data == "value"
                value = int(value_tree.children[0].value)

                system_params[name] = value

            elif tag == "pattern":
                raise NotImplementedError

            else:
                raise TypeError(f"Unsupported input type: {tag}")

        system = cls(None, rules, observables, variables)
        for init in inits:
            system.mixture.instantiate(init[1], int(init[0].evaluate(system)))
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

        Returns:
            A new System instance.
        """
        real_rules = []
        if rules is not None:
            for rule in rules:
                real_rules.extend(KappaRule.list_from_kappa(rule))

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

        return cls(
            None if mixture is None else Mixture.from_kappa(mixture),
            real_rules,
            real_observables,
            real_variables,
            *args,
            **kwargs,
        )

    def __init__(
        self,
        mixture: Optional[Mixture] = None,
        rules: Optional[Iterable[Rule]] = None,
        observables: Optional[dict[str, Expression]] = None,
        variables: Optional[dict[str, Expression]] = None,
        monitor: bool = True,
    ):
        """Initialize a new System.

        Args:
            mixture: Initial mixture state.
            rules: Collection of rules to apply.
            observables: Dictionary of observable expressions.
            variables: Dictionary of variable expressions.
            monitor: Whether to enable monitoring of simulation history.
        """
        self.rules = (
            {} if rules is None else {f"r{i}": rule for i, rule in enumerate(rules)}
        )

        if not isinstance(mixture, ComponentMixture) and any(
            type(rule) in [KappaRuleUnimolecular, KappaRuleBimolecular]
            for rule in self.rules.values()
        ):
            patterns = [] if mixture is None else [Pattern(list(mixture.agents))]
            mixture = ComponentMixture(patterns)

        if mixture is None:
            mixture = (
                ComponentMixture()
                if any(
                    type(rule) in [KappaRuleUnimolecular, KappaRuleBimolecular]
                    for rule in self.rules
                )
                else Mixture()
            )

        self.observables = {} if observables is None else observables
        self.variables = {} if variables is None else variables

        self.set_mixture(mixture)
        self.time = 0

        self.tallies = defaultdict(lambda: {"applied": 0, "failed": 0})
        if monitor:
            self.monitor = Monitor(self)
            self.monitor.update()
        else:
            self.monitor = None

    def __str__(self):
        return self.kappa_str

    def __getitem__(self, name: str) -> int | float:
        """Get the value of an observable or variable.

        Args:
            name: Name of the observable or variable.

        Returns:
            Current value of the named expression.

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

    def __setitem__(self, name: str, kappa_str: str):
        """Set or update an observable or variable from a Kappa string.

        Args:
            name: Name to assign to the expression.
            kappa_str: Kappa expression string.
        """
        expr = Expression.from_kappa(kappa_str)
        self._track_expression(expr)
        if name in self.variables:
            self.variables[name] = expr
        else:  # Set new expressions as observables
            self.observables[name] = expr

    @property
    def names(self) -> dict[str, set[str]]:
        """The names of all observables and variables."""
        return {
            "observables": set(self.observables),
            "variables": set(self.variables),
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

    @property
    def kappa_str(self) -> str:
        """The system representation in Kappa (.ka style) format."""
        kappa_str = ""
        for var_name, var in self.variables.items():
            kappa_str += f"%var: '{var_name}' {var.kappa_str}\n"
        for rule in self.rules.values():
            assert isinstance(rule, KappaRule)
            kappa_str += f"{rule.kappa_str}\n"
        for obs_name, obs in self.observables.items():
            obs_str = (
                f"|{obs.kappa_str}|" if isinstance(obs, Component) else obs.kappa_str
            )
            kappa_str += f"%obs: '{obs_name}' {obs_str}\n"
        kappa_str += self.mixture.kappa_str
        return kappa_str

    def to_ka(self, filepath: str) -> None:
        """Write system information to a Kappa file.

        Args:
            filepath: Path where to write the Kappa file.
        """
        with open(filepath, "w") as f:
            f.write(self.kappa_str)

    def set_mixture(self, mixture: Mixture) -> None:
        """Set the system's mixture and update tracking.

        Args:
            mixture: New mixture to set for the system.
        """
        self.mixture = mixture
        for rule in self.rules.values():
            self._track_rule(rule)
        for observable in self.observables.values():
            self._track_expression(observable)
        for variable in self.variables.values():
            self._track_expression(variable)

    def add_rule(self, rule: Rule | str, name: Optional[str] = None) -> None:
        """Add a new rule to the system.

        Args:
            rule: Rule object or Kappa string representation.
            name: Name to assign to the rule. If None, a default name is generated.

        Raises:
            AssertionError: If a rule with the given name already exists.
        """
        if name is None:
            name = f"r{len(self.rules)}"
            while name in self.rules:
                name = f"r{int(name[1:]) + 1}"
        assert name not in self.rules, "Rule {name} already exists in the system"

        if isinstance(rule, str):
            rule = KappaRule.from_kappa(rule)
        self._track_rule(rule)
        self.rules[name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove a rule by setting its rate to zero.

        Args:
            name: Name of the rule to remove.

        Raises:
            AssertionError: If the rule already has zero rate.
            KeyError: If no rule with the given name exists.
        """
        assert self.rules[name].rate(self) > 0, "Rule {name} is already null"
        try:
            self.rules[name].stochastic_rate = Expression.from_kappa("0")
        except KeyError as e:
            e.add_note("No rule {name} exists in the system")
            raise e

    def _track_rule(self, rule: Rule) -> None:
        """Track components mentioned in the left hand side of a Rule.

        Args:
            rule: Rule whose components should be tracked.
        """
        if isinstance(rule, KappaRule):
            for component in rule.left.components:
                # TODO: For efficiency check for isomorphism with already-tracked components
                self.mixture.track_component(component)

    def _track_expression(self, expression: Expression) -> None:
        """Track the Components in the given expression.

        Note:
            Doesn't track patterns nested by indirection - see the filter method.
        """
        for component_expr in expression.filter("component_pattern"):
            self.mixture.track_component(component_expr.attrs["value"])

    @cached_property
    def rule_reactivities(self) -> list[float]:
        """The reactivity of each rule in the system.

        Returns:
            List of reactivities corresponding to system rules.
        """
        return [rule.reactivity(self) for rule in self.rules.values()]

    @property
    def reactivity(self) -> float:
        """The total reactivity of the system.

        Returns:
            Sum of all rule reactivities.
        """
        return sum(self.rule_reactivities)

    def wait(self) -> None:
        """Advance simulation time according to exponential distribution.

        Raises:
            RuntimeWarning: If system has no reactivity (infinite wait time).
        """
        try:
            self.time += random.expovariate(self.reactivity)
        except ZeroDivisionError:
            warnings.warn(
                "system has no reactivity: infinite wait time", RuntimeWarning
            )

    def choose_rule(self) -> Optional[Rule]:
        """Choose a rule to apply based on reactivity weights.

        Returns:
            Selected rule, or None if no rules have positive reactivity.
        """
        try:
            return random.choices(
                list(self.rules.values()), weights=self.rule_reactivities
            )[0]
        except ValueError:
            warnings.warn("system has no reactivity: no rule applied", RuntimeWarning)
            return None

    def apply_rule(self, rule: Rule) -> None:
        """Apply a rule to the mixture and update tallies.

        Args:
            rule: Rule to apply to the current mixture.
        """
        update = rule.select(self.mixture)
        if update is not None:
            self.tallies[str(rule)]["applied"] += 1
            self.mixture.apply_update(update)
            del self.__dict__["rule_reactivities"]
        else:
            self.tallies[str(rule)]["failed"] += 1

    def _warn_about_rule_symmetries(self) -> None:
        if any(
            isinstance(rule, KappaRule) and rule.n_symmetries > 1
            for rule in self.rules.values()
        ):
            warnings.warn(
                "Some rules have multiple symmetries; PyKappa normalizes reactivities correspondingly. "
                "Results may differ from KaSim."
            )

    def update(self) -> None:
        """Perform one simulation step."""
        self._warn_about_rule_symmetries()
        self.wait()
        if (rule := self.choose_rule()) is not None:
            self.apply_rule(rule)
        if self.monitor:
            self.monitor.update()

    def update_via_kasim(self, time: float) -> None:
        """Simulate for a given amount of time using KaSim.

        Note:
            KaSim must be installed and in the PATH.
            Some features may not be compatible between PyKappa and KaSim.

        Args:
            time: Additional time units to simulate.

        Raises:
            AssertionError: If KaSim is not found in PATH.
        """
        self._warn_about_rule_symmetries()
        assert shutil.which(
            "KaSim"
        ), "To update via KaSim, it must be installed and in the PATH."

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Run KaSim on the current system
            output_ka_path = os.path.join(tmpdirname, "out.ka")
            output_cmd = f'%mod: alarm {time} do $SNAPSHOT "{output_ka_path}";'
            input_ka_path = os.path.join(tmpdirname, "in.ka")
            with open(input_ka_path, "w") as f:
                f.write(f"{self.kappa_str}\n{output_cmd}")
            os.system(f"KaSim {input_ka_path} -l {time} -d {tmpdirname}")

            # Read the KaSim output
            output_kappa_str = ""
            with open(output_ka_path) as f:
                for line in f:
                    if line.startswith("%init"):
                        split = line.split("/")
                        output_kappa_str += split[0] + split[-1]

        # Apply the update
        self.set_mixture(System.from_ka(output_kappa_str).mixture)
        self.time += time
        if self.monitor:
            self.monitor.update()


class Monitor:
    """Records the history of the values of observables in a system.

    Attributes:
        system: The system being monitored.
        history: Dictionary mapping observable names to their value history.
    """

    system: System
    history: dict[str, list[Optional[float]]]

    def __init__(self, system: System):
        """Initialize a monitor for the given system.

        Args:
            system: System to monitor.
        """
        self.system = system
        self.history = {"time": []} | {obs_name: [] for obs_name in system.observables}

    def update(self) -> None:
        """Record current time and observable values."""
        self.history["time"].append(self.system.time)
        for obs_name in self.system.observables:
            if obs_name not in self.history:
                self.history[obs_name] = [None] * (len(self.history["time"]) - 1)
            self.history[obs_name].append(self.system[obs_name])

    def measure(self, observable_name: str, time: Optional[float] = None):
        """Get the value of an observable at a specific time.

        Args:
            observable_name: Name of the observable to measure.
            time: Time at which to measure. If None, uses latest time.

        Returns:
            Value of the observable at the specified time.

        Raises:
            AssertionError: If simulation hasn't reached the specified time.
        """
        times: list[int] = list(self.history["time"])
        if time is None:
            time = times[-1]
        assert time <= max(times), "Simulation hasn't reached time {time}"

        i = 0
        while times[i] < time:
            i += 1
        return self.history[observable_name][i]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the history of observable values as a pandas DataFrame.

        Returns:
            DataFrame with time and observable columns.
        """
        return pd.DataFrame(self.history)

    def plot(self, combined: bool = False) -> matplotlib.figure.Figure:
        """Make a plot of all observables over time.

        Args:
            combined: Whether to plot all observables on the same axes.

        Returns:
            Matplotlib figure showing trajectories of observables.
        """
        if combined:
            fig, ax = plt.subplots()
            for obs_name in self.system.observables:
                ax.plot(self.history["time"], self.history[obs_name], label=obs_name)
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Observable")
            plt.margins(0, 0)
        else:
            fig, axs = plt.subplots(len(self.system.observables), 1, sharex=True)
            if len(self.system.observables) == 1:
                axs = [axs]
            for i, obs_name in enumerate(self.system.observables):
                axs[i].plot(self.history["time"], self.history[obs_name], color="black")
                axs[i].set_title(obs_name)
                if i == len(self.system.observables) - 1:
                    axs[i].set_xlabel("Time")
        return fig
