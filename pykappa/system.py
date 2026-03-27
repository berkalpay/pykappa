import os
import shutil
import tempfile
import random
import warnings
from collections import defaultdict
from functools import cached_property
from typing import Optional, Iterable, Self
from graphviz import Source

from pykappa.mixture import Mixture
from pykappa.rule import Rule, UnimolecularRule, BimolecularRule
from pykappa.pattern import Component, Pattern
from pykappa.analysis import Monitor
from pykappa._expression import Expression
from pykappa._utils import str_table


class System:
    """A Kappa system containing agents, rules, observables, and variables for simulation."""

    mixture: Mixture  #: The current state of agents and their connections
    rules: dict[str, Rule]  #: Maps rule names to Rule objects
    observables: dict[str, Expression]  #: Maps observable names to expressions
    variables: dict[str, Expression]  #: Maps variable names to expressions
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
        rules: list[Rule] = []
        system_params: dict[str, int] = {}
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
                raise NotImplementedError

            elif tag == "init_declaration":
                expr_tree = child.children[0]
                assert expr_tree.data == "algebraic_expression"
                amount = ExpressionTransformer.from_tree(expr_tree)

                pattern_tree = child.children[1]
                if pattern_tree.data == "declared_token_name":
                    raise NotImplementedError
                assert pattern_tree.data == "pattern"
                pattern = KappaTransformer().transform(pattern_tree)

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

        system = cls(None, rules, observables, variables, seed=seed)
        for init in inits:
            system.mixture.add(init[1], int(init[0].evaluate(system)))
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
        seed: Optional[int] = None,
    ):
        """
        Args:
            mixture: Initial mixture state.
            rules: Collection of rules to apply.
            observables: Dictionary of observable expressions.
            variables: Dictionary of variable expressions.
            monitor: Whether to enable monitoring of simulation history.
            seed: Random seed for reproducibility.
        """
        self._rng = random.Random() if seed is None else random.Random(seed)

        self.rules = (
            {} if rules is None else {f"r{i}": rule for i, rule in enumerate(rules)}
        )

        mixture = Mixture() if mixture is None else mixture
        if any(
            type(rule) in [UnimolecularRule, BimolecularRule]
            for rule in self.rules.values()
        ):
            mixture.enable_component_tracking()

        self.observables = {} if observables is None else observables
        self.variables = {} if variables is None else variables

        self._set_mixture(mixture)
        self.time = 0

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
            i = 0
            while (name := f"r{i}") in self.rules:
                i += 1
        assert name not in self.rules, "Rule {name} already exists in the system"

        if isinstance(rule, str):
            rule = Rule.from_kappa(rule)

        if type(rule) in [UnimolecularRule, BimolecularRule]:
            self.mixture.enable_component_tracking()

        self._track_rule(rule)
        self.rules[name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove a rule by setting its rate to zero.

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
        """Track components mentioned in the left hand side of a Rule."""
        for component in rule.left.components:
            # TODO: For efficiency check for isomorphism with already-tracked components
            self.mixture._track_component(component)

    def _track_expression(self, expression: Expression) -> None:
        """Track the Components in the given expression.

        Note:
            Doesn't track patterns nested by indirection - see the filter method.
        """
        for component_expr in expression.filter("component_pattern"):
            self.mixture._track_component(component_expr.attrs["value"])

    @cached_property
    def rule_reactivities(self) -> list[float]:
        """The reactivity of each rule in the system."""
        return [rule.reactivity(self) for rule in self.rules.values()]

    @property
    def reactivity(self) -> float:
        """The total reactivity of the system."""
        return sum(self.rule_reactivities)

    def wait(self) -> None:
        """Advance simulation time according to exponential distribution.

        Raises:
            RuntimeWarning: If system has no reactivity (infinite wait time).
        """
        try:
            self.time += self._rng.expovariate(self.reactivity)
        except ZeroDivisionError:
            warnings.warn(
                "system has no reactivity: infinite wait time", RuntimeWarning
            )

    def _choose_rule(self) -> Optional[Rule]:
        """Choose a rule to apply based on reactivity weights.

        Returns:
            Selected rule, or None if no rules have positive reactivity.
        """
        try:
            return self._rng.choices(
                list(self.rules.values()), weights=self.rule_reactivities
            )[0]
        except ValueError:
            warnings.warn("system has no reactivity: no rule applied", RuntimeWarning)
            return None

    def _apply_rule(self, rule: Rule) -> None:
        """Apply a rule to the mixture and update tallies."""
        update = rule._select(self.mixture)
        if update is not None:
            self.tallies[str(rule)]["applied"] += 1
            self.mixture._apply_update(update)
            del self.__dict__["rule_reactivities"]
        else:
            self.tallies[str(rule)]["failed"] += 1

    def update(self) -> None:
        """Perform one simulation step."""
        if self.monitor is not None and not self.monitor.history["time"]:
            self.monitor.update()  # Record initial state

        self.wait()
        if (rule := self._choose_rule()) is not None:
            self._apply_rule(rule)

        if self.monitor is not None:
            self.monitor.update()

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
        self._set_mixture(System.from_ka(output_kappa_str).mixture)
        self.time += time
        if self.monitor:
            self.monitor.update()

    def contact_map(self):
        from pykappa.analysis import contact_map

        return contact_map(self)
