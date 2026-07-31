import math
import operator
from collections import deque
from typing import Any, Self, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pykappa.pattern import Component
    from pykappa.system import System


_string_to_operator = {
    # Unary
    "[log]": math.log,
    "[exp]": math.exp,
    "[sin]": math.sin,
    "[cos]": math.cos,
    "[tan]": math.tan,
    "[sqrt]": math.sqrt,
    # Binary
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
    "mod": operator.mod,
    # Comparisons
    "=": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
    # List
    "[max]": max,
    "[min]": min,
}


def _parse_operator(kappa_operator: str) -> Callable:
    """Convert a Kappa string operator to a Python function.

    Raises:
        ValueError: If the operator is not recognized.
    """
    try:
        return _string_to_operator[kappa_operator]
    except KeyError:
        raise ValueError(f"Unknown operator: {kappa_operator}")


class Expression:
    """Algebraic expressions as specified by the Kappa language."""

    _type: Any  # Type of expression (literal, variable, binary_op, etc.)
    _attrs: dict[str, Any]  # Dictionary of attributes specific to the expression type

    @classmethod
    def from_kappa(cls, kappa_str: str) -> "Expression":
        """Parse an Expression from a Kappa string.

        Raises:
            AssertionError: If the string doesn't represent a valid expression.
        """
        from pykappa._parsing import kappa_parser, ExpressionTransformer

        input_tree = kappa_parser.parse(kappa_str)
        assert input_tree.data == "kappa_input"
        expr_tree = input_tree.children[0]
        assert expr_tree.data in ["!algebraic_expression", "algebraic_expression"]
        return ExpressionTransformer.from_tree(expr_tree)

    def __init__(self, type, **attrs):
        self._type = type
        self._attrs = attrs

    def __str__(self):
        return self.kappa_str

    @property
    def kappa_str(self) -> str:
        """
        Raises:
            ValueError: If expression type is not supported for string conversion.
        """
        if self._type == "literal":
            return str(self.evaluate())
        if self._type == "boolean_literal":
            return "[true]" if self._attrs["value"] else "[false]"
        if self._type == "variable":
            return f"'{self._attrs['name']}'"
        if self._type in ("binary_op", "comparison", "logical_or", "logical_and"):
            operator = {
                "logical_or": "||",
                "logical_and": "&&",
            }.get(self._type, self._attrs.get("operator"))
            return f"({self._attrs['left'].kappa_str}) {operator} ({self._attrs['right'].kappa_str})"
        if self._type in ("unary_op", "logical_not"):
            operator = "[not]" if self._type == "logical_not" else self._attrs["operator"]
            return f"{operator} ({self._attrs['child'].kappa_str})"
        if self._type == "list_op":
            children = " ".join(
                f"({child.kappa_str})" for child in self._attrs["children"]
            )
            return f"{self._attrs['operator']} {children}"
        if self._type == "defined_constant":
            return self._attrs["name"]
        if self._type == "parentheses":
            return self._attrs["child"].kappa_str
        if self._type == "conditional":
            return (
                f"{self._attrs['condition'].kappa_str} [?] {self._attrs['true_expr'].kappa_str} "
                f"[:] {self._attrs['false_expr'].kappa_str}"
            )
        if self._type == "reserved_variable":
            return self._attrs["value"].kappa_str
        if self._type == "component_pattern":
            return f"|{self._attrs['value'].kappa_str}|"
        if self._type == "token_value":
            return f"|{self._attrs['name']}|"

        raise ValueError(f"Unsupported node type: {self._type}")

    def evaluate(self, system: Optional["System"] = None) -> int | float:
        """Evaluate the expression to get its value.

        Args:
            system: System context for variable evaluation (required for variables).

        Raises:
            ValueError: If evaluation fails due to missing context or unsupported type.
        """
        if self._type in ("literal", "boolean_literal"):
            return self._attrs["value"]
        if self._type == "variable":
            name = self._attrs["name"]
            if system is None:
                raise ValueError(f"{self} needs a System to evaluate variable '{name}'")
            return system[name]
        if self._type in ("binary_op", "comparison", "logical_or", "logical_and"):
            left = self._attrs["left"].evaluate(system)
            right = self._attrs["right"].evaluate(system)
            if self._type == "logical_or":
                return left or right
            if self._type == "logical_and":
                return left and right
            return _parse_operator(self._attrs["operator"])(left, right)
        if self._type in ("unary_op", "logical_not"):
            child = self._attrs["child"].evaluate(system)
            return (
                not child
                if self._type == "logical_not"
                else _parse_operator(self._attrs["operator"])(child)
            )
        if self._type == "list_op":
            children = [child.evaluate(system) for child in self._attrs["children"]]
            return _parse_operator(self._attrs["operator"])(children)
        if self._type == "defined_constant":
            const = self._attrs["name"]
            if const == "[pi]":
                return math.pi
            raise ValueError(f"Unknown constant: {const}")
        if self._type == "parentheses":
            return self._attrs["child"].evaluate(system)
        if self._type == "conditional":
            return (
                self._attrs["true_expr"].evaluate(system)
                if self._attrs["condition"].evaluate(system)
                else self._attrs["false_expr"].evaluate(system)
            )
        if self._type == "reserved_variable":
            value = self._attrs["value"]
            if value._type == "component_pattern":
                component: Component = value._attrs["value"]
                if system is None:
                    raise ValueError(
                        f"{self} needs a System to evaluate pattern {component}"
                    )
                return (
                    len(system.mixture.embeddings(component))
                    // value._attrs["n_symmetries"]
                )
            raise NotImplementedError(
                f"Reserved variable {value._type} not implemented yet."
            )
        if self._type == "token_value":
            name = self._attrs["name"]
            if system is None:
                raise ValueError(f"{self} needs a System to evaluate token '{name}'")
            return system.tokens.get(name, 0.0)

        raise ValueError(f"Unsupported node type: {self._type}")

    def _filter(self, type_str: str) -> list[Self]:
        """
        Returns all nodes in the expression tree whose type matches the provided string.

        Note:
            Doesn't detect nodes indirectly nested in named variables.
        """
        result = []
        stack = deque([self])  # DFS from the root

        while stack:
            node = stack.pop()
            if node._type == type_str:
                result.append(node)

            # Add child nodes to the stack
            if hasattr(node, "_attrs"):
                for attr_value in node._attrs.values():
                    if isinstance(attr_value, type(self)):
                        stack.append(attr_value)
                    elif isinstance(attr_value, (list, tuple)):
                        stack.extend(v for v in attr_value if isinstance(v, type(self)))

        return result
